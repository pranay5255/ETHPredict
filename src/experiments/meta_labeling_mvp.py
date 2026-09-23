"""MVP hierarchical meta-labeling research pipeline.

This module implements the v2 config path used by ``configs/config.yml``. It is
kept separate from ``staged_trial.py`` so legacy staged GLFT experiments remain
runnable while the active config moves to multi-horizon forecasting, true
meta-labels, and directional alpha backtesting.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.config.loader import expand_grid_search
from src.data.features_all import DEFAULT_GRANULARITY, DataPreprocessor, bars_for_duration
from src.features.labeling import meta_triple_barrier_labels
from src.features.sample_weights import (
    average_uniqueness,
    build_label_spans,
    sample_weight_config,
    sample_weight_diagnostics,
    sample_weights_and_diagnostics,
)
from src.training.devices import resolve_training_device
from src.training.trainer import compute_metrics
from src.utils.trackio_logging import enforce_trackio_policy, log_trackio_run


class MultiHorizonLSTM(nn.Module):
    """Shared LSTM encoder with return and direction heads per horizon."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, n_horizons: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_horizons = n_horizons
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.return_head = nn.Linear(hidden_size, n_horizons)
        self.direction_head = nn.Linear(hidden_size, n_horizons)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(x)
        last = encoded[:, -1, :]
        return self.return_head(last), self.direction_head(last)


class MetaLabelMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_size
        for _ in range(max(1, num_layers)):
            layers.extend([nn.Linear(prev, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden_size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ConstantMetaModel:
    def __init__(self, probability: float, feature_columns: Sequence[str]):
        self.probability = float(np.clip(probability, 0.0, 1.0))
        self.feature_columns = list(feature_columns)
        self.kind = "constant"

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.probability, dtype=float)

    def manifest(self) -> Dict[str, Any]:
        return {"kind": self.kind, "probability": self.probability, "feature_columns": self.feature_columns}


class TorchMetaModel:
    def __init__(self, model: MetaLabelMLP, mean: np.ndarray, std: np.ndarray, feature_columns: Sequence[str], device: torch.device):
        self.model = model
        self.mean = mean
        self.std = std
        self.feature_columns = list(feature_columns)
        self.device = device
        self.kind = "mlp"

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame.reindex(columns=self.feature_columns).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        return (X - self.mean) / self.std

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.array([], dtype=float)
        self.model.eval()
        X = torch.tensor(self._matrix(frame), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return torch.sigmoid(self.model(X)).detach().cpu().numpy().reshape(-1)

    def manifest(self) -> Dict[str, Any]:
        return {"kind": self.kind, "feature_columns": self.feature_columns}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_json_ready(payload), sort_keys=False), encoding="utf-8")
    return path


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(args: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_manifest() -> Dict[str, Any]:
    status = _command_output(["git", "status", "--short"])
    critical = []
    for line in (status or "").splitlines():
        path = line[3:] if len(line) > 3 else line
        if path.startswith(("configs/", "src/", "tests/", "context/", "TASKS.md")):
            critical.append({"status": line[:2].strip(), "path": path})
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status or "",
        "critical_untracked_or_modified": critical,
    }


def _dependency_manifest() -> Dict[str, Any]:
    packages = ["numpy", "pandas", "torch", "pyarrow", "scikit-learn", "trackio", "PyYAML"]
    versions: Dict[str, Optional[str]] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _raw_data_manifest(config: Mapping[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {}) or {}
    granularity = data_cfg.get("granularity", data_cfg.get("lighter", {}).get("resolution", DEFAULT_GRANULARITY))
    raw_dir = Path(data_cfg.get("dir", "data")) / "raw"
    files = sorted(raw_dir.glob(f"ETHUSDT-{granularity}-lighter-*.csv"))
    return {
        "raw_dir": raw_dir,
        "granularity": granularity,
        "files": [
            {"path": path, "bytes": path.stat().st_size, "sha256": _sha256_path(path)}
            for path in files
        ],
    }


def _device_manifest(device: torch.device) -> Dict[str, Any]:
    return {
        "requested": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else None,
    }


def _config_identity(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    raw = _load_yaml(config_path) if config_path.exists() else {}
    return {
        "resolved_config_path": config_path,
        "resolved_config_hash": _stable_hash(config),
        "source_config_hash": _stable_hash(raw),
    }


def _record_final_test_evaluation(config: Mapping[str, Any], run_dir: Path, *, trial_id: str, split_hash: str, smoke: bool) -> Dict[str, Any]:
    guard_cfg = ((config.get("research", {}) or {}).get("final_test_guard", {}) or {})
    spec_hash = _stable_hash({"config": config, "split_hash": split_hash})
    ledger_path = run_dir.parent / "_final_test_reuse_ledger.json"
    ledger: Dict[str, Any] = {}
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            ledger = {}
    record = dict(ledger.get(spec_hash, {}))
    count = int(record.get("count", 0)) + 1
    blocked = bool(guard_cfg.get("do_not_reuse_test", config.get("do_not_reuse_test", False))) and count > 1 and not smoke
    record.update({"count": count, "last_trial_id": trial_id, "last_run_dir": str(run_dir), "split_hash": split_hash})
    ledger[spec_hash] = record
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    payload = {"spec_hash": spec_hash, "count": count, "ledger_path": ledger_path, "blocked": blocked}
    if blocked:
        raise RuntimeError(f"Final test set reuse blocked for research spec {spec_hash}; ledger count={count}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _sanitize_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    return {str(key): _safe_float(value) for key, value in metrics.items()}


def _slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in value.strip()).strip("-")
    return out or "run"


def _run_id(run_name: str) -> str:
    return f"{_slug(run_name)}_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _environment_manifest() -> Dict[str, Any]:
    return {
        "created_at_utc": _utc_now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "git": _git_manifest(),
        "dependencies": _dependency_manifest(),
    }


def apply_smoke_overrides(config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(dict(config))
    smoke = cfg.get("smoke", {}) or {}
    if "max_trials" in smoke:
        cfg.setdefault("search", {})["max_trials"] = int(smoke["max_trials"])
    if "folds" in smoke:
        cfg.setdefault("validation", {})["folds"] = int(smoke["folds"])
    training = cfg.setdefault("training", {})
    for key in ["sequence_length", "epochs", "batch_size"]:
        if key in smoke:
            training[key] = smoke[key]
    model_base = cfg.setdefault("model", {}).setdefault("base", {})
    meta = cfg.setdefault("model", {}).setdefault("meta_labeler", {})
    if "hidden_size" in smoke:
        model_base["hidden_size"] = smoke["hidden_size"]
        meta["hidden_size"] = smoke["hidden_size"]
    return cfg


def target_horizons(config: Mapping[str, Any]) -> List[Tuple[str, int]]:
    horizons = config.get("targets", {}).get("horizons", {})
    if not horizons:
        raise ValueError("v2 config requires targets.horizons")
    return [(str(name), int(spec["bars"])) for name, spec in horizons.items()]


def _normalise_features(features_df: pd.DataFrame) -> torch.Tensor:
    features = torch.tensor(features_df.to_numpy(dtype=float), dtype=torch.float32)
    mean = features.mean(dim=0)
    std = features.std(dim=0) + 1e-8
    return (features - mean) / std


def _total_cost_bps(config: Mapping[str, Any], horizon_bars: int, granularity: str) -> float:
    costs = config.get("costs", {}) or {}
    bars_per_hour = bars_for_duration(granularity, hours=1)
    horizon_hours = horizon_bars / max(bars_per_hour, 1)
    return (
        2.0 * float(costs.get("fee_bps", 0.0))
        + float(costs.get("spread_bps", 0.0))
        + float(costs.get("slippage_bps", 0.0))
        + float(costs.get("funding_bps_per_hour", 0.0)) * horizon_hours
    )


def _granularity_timedelta(granularity: str, bars: int = 1) -> pd.Timedelta:
    raw = str(granularity).strip().lower()
    try:
        value = int(raw[:-1])
        unit = raw[-1]
    except (TypeError, ValueError, IndexError):
        return pd.Timedelta(minutes=5 * max(1, int(bars)))
    if unit == "m":
        return pd.Timedelta(minutes=value * max(1, int(bars)))
    if unit == "h":
        return pd.Timedelta(hours=value * max(1, int(bars)))
    if unit == "d":
        return pd.Timedelta(days=value * max(1, int(bars)))
    return pd.Timedelta(minutes=5 * max(1, int(bars)))


def _sampling_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config.get("sampling", config.get("events", {})) if isinstance(config, Mapping) else {}
    return dict(raw or {})


def _distribution_summary(series: pd.Series) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "p05": float(clean.quantile(0.05)),
        "p50": float(clean.quantile(0.50)),
        "p95": float(clean.quantile(0.95)),
    }


def _span_uniqueness(indices: Sequence[int], horizon_bars: int) -> Dict[str, float]:
    spans = [(int(idx), int(idx) + max(1, int(horizon_bars))) for idx in indices]
    if not spans:
        return {"average_uniqueness": 0.0, "effective_sample_size": 0.0}
    start = min(left for left, _ in spans)
    end = max(right for _, right in spans)
    concurrency = np.zeros(end - start + 1, dtype=float)
    for left, right in spans:
        concurrency[left - start : right - start + 1] += 1.0
    uniqueness = []
    for left, right in spans:
        active = concurrency[left - start : right - start + 1]
        uniqueness.append(float(np.mean(1.0 / np.maximum(active, 1.0))))
    return {
        "average_uniqueness": float(np.mean(uniqueness)),
        "effective_sample_size": float(np.sum(uniqueness)),
    }


def _bar_clock_diagnostics(
    features_df: pd.DataFrame,
    log_returns: pd.Series,
    realized_vol: pd.Series,
    y_dir_array: np.ndarray,
    horizons: Sequence[Tuple[str, int]],
) -> Dict[str, Any]:
    missingness = {
        str(column): float(value)
        for column, value in features_df.isna().mean().sort_values(ascending=False).head(25).items()
        if float(value) > 0.0
    }
    return {
        "missingness_top": missingness,
        "serial_correlation": {
            "lag_1_log_return": _safe_float(log_returns.autocorr(lag=1), 0.0),
            "lag_12_log_return": _safe_float(log_returns.autocorr(lag=min(12, max(1, len(log_returns) - 1))), 0.0),
        },
        "realized_volatility_distribution": _distribution_summary(realized_vol),
        "log_return_distribution": _distribution_summary(log_returns),
        "label_balance": {
            name: float(y_dir_array[:, idx].mean()) if len(y_dir_array) else 0.0
            for idx, (name, _) in enumerate(horizons)
        },
    }


def _select_event_indices(
    log_returns: pd.Series,
    realized_vol: pd.Series,
    features_df: pd.DataFrame,
    dense_indices: Sequence[int],
    config: Mapping[str, Any],
) -> Tuple[List[int], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    sampling = _sampling_config(config)
    mode = str(sampling.get("mode", "dense")).lower()
    dense = [int(idx) for idx in dense_indices]
    if mode in {"dense", "rolling"}:
        metadata = {idx: {"event_time_index": idx, "trigger_type": "dense", "trigger_threshold": 0.0, "reference_volatility": float(realized_vol.iloc[idx])} for idx in dense}
        return dense, metadata, {"mode": "dense", "dense_observations": len(dense), "events": len(dense), "event_rate": 1.0}
    if mode in {"model_edge", "edge", "edge_triggered"}:
        edge_column = str(sampling.get("edge_column", "log_return"))
        if edge_column not in features_df.columns:
            raise ValueError(f"Edge-triggered observation sampling requires feature column {edge_column!r}")
        threshold = float(sampling.get("edge_threshold_bps", sampling.get("threshold_bps", 10.0))) / 10_000.0
        selected = [idx for idx in dense if abs(float(features_df.iloc[idx][edge_column])) >= threshold]
        metadata = {
            idx: {
                "event_time_index": idx,
                "trigger_type": "edge_triggered",
                "trigger_threshold": float(threshold),
                "reference_volatility": float(realized_vol.iloc[idx]),
                "edge_column": edge_column,
            }
            for idx in selected
        }
        return selected, metadata, {
            "mode": mode,
            "dense_observations": len(dense),
            "events": len(selected),
            "event_rate": float(len(selected) / max(len(dense), 1)),
            "edge_column": edge_column,
            "edge_threshold_bps": float(threshold * 10_000.0),
        }

    fixed_threshold = float(sampling.get("threshold_bps", sampling.get("cusum_threshold_bps", 10.0))) / 10_000.0
    vol_multiplier = float(sampling.get("volatility_multiplier", sampling.get("cusum_volatility_multiplier", 1.0)))
    min_threshold = float(sampling.get("min_threshold_bps", 1.0)) / 10_000.0
    selected: List[int] = []
    metadata: Dict[int, Dict[str, Any]] = {}
    s_pos = 0.0
    s_neg = 0.0
    dense_set = set(dense)
    for idx in dense:
        value = float(log_returns.iloc[idx])
        s_pos = max(0.0, s_pos + value)
        s_neg = min(0.0, s_neg + value)
        ref_vol = float(realized_vol.iloc[idx])
        threshold = max(min_threshold, ref_vol * vol_multiplier) if mode in {"volatility_cusum", "vol_scaled_cusum", "volatility-scaled-cusum"} else fixed_threshold
        trigger = None
        if s_pos > threshold:
            trigger = "positive_cusum"
        elif abs(s_neg) > threshold:
            trigger = "negative_cusum"
        if trigger and idx in dense_set:
            selected.append(idx)
            metadata[idx] = {
                "event_time_index": idx,
                "trigger_type": trigger,
                "trigger_threshold": float(threshold),
                "reference_volatility": ref_vol,
            }
            s_pos = 0.0
            s_neg = 0.0
    diagnostics = {
        "mode": mode,
        "dense_observations": len(dense),
        "events": len(selected),
        "event_rate": float(len(selected) / max(len(dense), 1)),
        "threshold_bps": float(fixed_threshold * 10_000.0),
        "volatility_multiplier": vol_multiplier,
    }
    return selected, metadata, diagnostics


def build_multi_horizon_lighter_dataset(config: Mapping[str, Any], *, smoke: bool = False) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    granularity = data_cfg.get("granularity", data_cfg.get("lighter", {}).get("resolution", DEFAULT_GRANULARITY))
    training_cfg = config.get("training", {})
    sequence_length = int(training_cfg.get("sequence_length", bars_for_duration(granularity, hours=24)))
    horizons = target_horizons(config)
    max_horizon = max(bars for _, bars in horizons)
    label_cfg = config.get("labels", {}).get("meta_triple_barrier", {})
    volatility_window = int(label_cfg.get("volatility_window", bars_for_duration(granularity, hours=24)))

    bar_cfg = dict(config.get("bars", {}) or data_cfg.get("bars", {}) or {})
    if data_cfg.get("bar_type") is not None and "type" not in bar_cfg:
        bar_cfg["type"] = data_cfg.get("bar_type")
    preprocessor = DataPreprocessor(
        data_dir=data_cfg.get("dir", "data"),
        granularities=[granularity],
        feature_config=config.get("features", {}) or {},
        bar_config=bar_cfg,
    )
    features_df, targets_df = preprocessor.get_base_dataset(granularity=granularity)

    if smoke:
        max_rows = int(config.get("smoke", {}).get("max_rows", 0) or 0)
        if max_rows:
            min_rows = sequence_length + max_horizon + 64
            keep_rows = max(max_rows, min_rows)
            features_df = features_df.tail(keep_rows)
            targets_df = targets_df.tail(keep_rows)

    if len(features_df) <= sequence_length + max_horizon:
        raise ValueError(
            f"Need more than sequence_length + max_horizon rows; got {len(features_df)}, "
            f"sequence_length={sequence_length}, max_horizon={max_horizon}"
        )

    prices = targets_df["close"].astype(float)
    log_returns = np.log(prices.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    realized_vol = log_returns.rolling(volatility_window, min_periods=2).std()
    fill_vol = _safe_float(realized_vol.mean(), 0.001)
    realized_vol = realized_vol.fillna(fill_vol).clip(lower=1e-8)

    raw_features = torch.tensor(features_df.to_numpy(dtype=float), dtype=torch.float32)
    X: List[torch.Tensor] = []
    y_ret: List[List[float]] = []
    y_dir: List[List[float]] = []
    sample_rows: List[Dict[str, Any]] = []
    dense_label_indices = list(range(sequence_length, len(features_df) - max_horizon))
    event_indices, event_metadata, event_diagnostics = _select_event_indices(log_returns, realized_vol, features_df, dense_label_indices, config)
    if not event_indices:
        raise ValueError(f"Sampling mode {event_diagnostics.get('mode')} produced no events")
    for label_idx in event_indices:
        start = label_idx - sequence_length
        X.append(raw_features[start:label_idx])
        ret_row: List[float] = []
        dir_row: List[float] = []
        entry = float(prices.iloc[label_idx])
        for _, horizon_bars in horizons:
            future = float(prices.iloc[label_idx + horizon_bars])
            ret = float(np.log(future / entry)) if entry > 0 and future > 0 else 0.0
            ret_row.append(ret)
            dir_row.append(1.0 if ret > 0 else 0.0)
        y_ret.append(ret_row)
        y_dir.append(dir_row)
        feature_row = features_df.iloc[label_idx]
        horizon_span_meta: Dict[str, Any] = {}
        for horizon_name, horizon_bars in horizons:
            horizon_end_idx = int(label_idx + horizon_bars)
            horizon_span_meta[f"{horizon_name}_label_span_end_idx"] = horizon_end_idx
            horizon_span_meta[f"{horizon_name}_label_span_end_timestamp"] = prices.index[horizon_end_idx]
        sample_rows.append(
            {
                "timestamp": prices.index[label_idx],
                "sample_index": label_idx,
                "close": entry,
                "open": float(feature_row.get("open", entry)),
                "high": float(feature_row.get("high", entry)),
                "low": float(feature_row.get("low", entry)),
                "volume": float(feature_row.get("volume", 0.0)),
                "realized_vol": float(realized_vol.iloc[label_idx]),
                "vol_regime": float(feature_row.get("vol_regime", 0.0)),
                "event_trigger_type": event_metadata.get(label_idx, {}).get("trigger_type", "unknown"),
                "event_trigger_threshold": event_metadata.get(label_idx, {}).get("trigger_threshold", 0.0),
                "event_reference_volatility": event_metadata.get(label_idx, {}).get("reference_volatility", float(realized_vol.iloc[label_idx])),
                "event_time": prices.index[label_idx],
                **horizon_span_meta,
                "source_bar_index": label_idx,
            }
        )

    path_columns = [col for col in ["open", "high", "low", "close", "volume"] if col in features_df.columns]
    price_path = features_df[path_columns].copy()
    price_path["timestamp"] = features_df.index
    price_path["realized_vol"] = realized_vol.to_numpy(dtype=float)
    if "vol_regime" in features_df.columns:
        price_path["vol_regime"] = features_df["vol_regime"].to_numpy(dtype=float)
    price_path = price_path.reset_index(drop=True)

    sample_df = pd.DataFrame(sample_rows)
    span_frame = build_label_spans(
        sample_df["sample_index"],
        pd.to_numeric(sample_df["sample_index"], errors="coerce") + max_horizon,
        timestamps=prices.index,
    )
    for column in span_frame.columns:
        sample_df[column] = span_frame[column].to_numpy()
    sample_df["average_uniqueness"] = average_uniqueness(sample_df["label_span_start_idx"], sample_df["label_span_end_idx"])

    weight_cfg = sample_weight_config(config)
    sample_weights_series, sample_weight_diag = sample_weights_and_diagnostics(
        sample_df,
        mode=weight_cfg["base_mode"],
        normalize=weight_cfg["normalize"],
        return_values=np.asarray(y_ret, dtype=float),
    )
    sample_df["sample_weight"] = sample_weights_series.to_numpy(dtype=float)
    sample_weight_diag = {
        **sample_weight_diag,
        "base_mode": weight_cfg["base_mode"],
        "meta_mode": weight_cfg["meta_mode"],
    }
    sample_weights = torch.tensor(sample_df["sample_weight"].to_numpy(dtype=float), dtype=torch.float32)
    y_dir_array = np.asarray(y_dir, dtype=float) if y_dir else np.empty((0, len(horizons)))
    event_diagnostics = dict(event_diagnostics)
    event_diagnostics["class_balance"] = {
        name: float(y_dir_array[:, idx].mean()) if len(y_dir_array) else 0.0
        for idx, (name, _) in enumerate(horizons)
    }
    event_diagnostics.update({
        "average_uniqueness": sample_weight_diag["average_uniqueness"],
        "effective_sample_size": sample_weight_diag["effective_sample_size"],
    })
    bar_clock_diagnostics = _bar_clock_diagnostics(features_df, log_returns, realized_vol, y_dir_array, horizons)
    return {
        "target": "multi_horizon",
        "X": torch.stack(X),
        "y_ret": torch.tensor(y_ret, dtype=torch.float32),
        "y_dir": torch.tensor(y_dir, dtype=torch.float32),
        "sample_weights": sample_weights,
        "samples": sample_df,
        "price_path": price_path,
        "close": pd.Series(sample_df["close"].to_numpy(dtype=float), index=pd.Index(sample_df["timestamp"], name="timestamp")),
        "input_size": int(raw_features.shape[1]),
        "granularity": granularity,
        "sequence_length": sequence_length,
        "horizon_names": [name for name, _ in horizons],
        "horizon_bars": {name: bars for name, bars in horizons},
        "max_horizon_bars": max_horizon,
        "feature_columns": preprocessor.get_feature_cols(),
        "feature_window_bars": preprocessor.feature_window_bars(granularity),
        "feature_manifest": preprocessor.feature_manifest(),
        "bar_manifest": preprocessor.bar_manifest(),
        "event_diagnostics": event_diagnostics,
        "sample_weight_diagnostics": sample_weight_diag,
        "bar_clock_diagnostics": bar_clock_diagnostics,
        "side_data_manifest": {"enabled_groups": [], "coverage": {}, "status": "disabled"},
    }


def purged_walk_forward_splits(n_samples: int, validation_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if n_samples < 8:
        raise ValueError(f"Need at least 8 samples for purged walk-forward; got {n_samples}")
    folds = int(validation_cfg.get("folds", 3))
    test_fraction = float(validation_cfg.get("test_fraction", 0.2))
    purge = int(validation_cfg.get("purge_bars", 0) or 0)
    embargo = int(validation_cfg.get("embargo_bars", 0) or 0)
    test_size = max(1, int(round(n_samples * test_fraction)))
    test_start = max(1, n_samples - test_size)
    dev_end = max(1, test_start - purge - embargo)
    if dev_end < folds + 2:
        dev_end = max(1, test_start - purge)
    fold_size = max(1, dev_end // (folds + 1))
    fold_specs: List[Dict[str, np.ndarray]] = []
    for fold_idx in range(folds):
        val_start = min(dev_end - 1, (fold_idx + 1) * fold_size)
        val_end = dev_end if fold_idx == folds - 1 else min(dev_end, val_start + fold_size)
        train_end = max(0, val_start - purge)
        train = np.arange(0, train_end, dtype=int)
        validation = np.arange(val_start, val_end, dtype=int)
        if len(train) and len(validation):
            fold_specs.append({"fold": fold_idx, "train": train, "validation": validation})
    if not fold_specs:
        split = max(1, dev_end // 2)
        fold_specs.append(
            {
                "fold": 0,
                "train": np.arange(0, split, dtype=int),
                "validation": np.arange(split, dev_end, dtype=int),
            }
        )
    return {
        "folds": fold_specs,
        "development": np.arange(0, dev_end, dtype=int),
        "gap": np.arange(dev_end, test_start, dtype=int),
        "test": np.arange(test_start, n_samples, dtype=int),
        "purge_bars": purge,
        "embargo_bars": embargo,
        "test_start": test_start,
        "development_end": dev_end,
    }


def _subset(dataset: Mapping[str, Any], indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return dataset["X"][indices], dataset["y_ret"][indices], dataset["y_dir"][indices], dataset["sample_weights"][indices]


def train_base_model(dataset: Mapping[str, Any], indices: np.ndarray, config: Mapping[str, Any], device: torch.device) -> Tuple[MultiHorizonLSTM, Dict[str, Any]]:
    X, y_ret, y_dir, weights = _subset(dataset, indices)
    X = X.to(device)
    feature_mean = X.mean(dim=(0, 1), keepdim=True)
    feature_std = X.std(dim=(0, 1), keepdim=True, unbiased=False) + 1e-8
    X = (X - feature_mean) / feature_std
    y_ret = y_ret.to(device)
    y_dir = y_dir.to(device)
    weights = weights.to(device)
    model_cfg = config.get("model", {}).get("base", {})
    training = config.get("training", {})
    loss_weights = training.get("loss_weights", {}) or {}
    model = MultiHorizonLSTM(
        input_size=int(dataset["input_size"]),
        hidden_size=int(model_cfg.get("hidden_size", 16)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        n_horizons=len(dataset["horizon_names"]),
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    ret_loss = nn.MSELoss(reduction="none")
    dir_loss = nn.BCEWithLogitsLoss(reduction="none")
    loader = DataLoader(
        TensorDataset(X, y_ret, y_dir, weights),
        batch_size=int(training.get("batch_size", 32)),
        shuffle=True,
    )
    losses: List[float] = []
    for _ in range(int(training.get("epochs", 1))):
        model.train()
        total = 0.0
        batches = 0
        for batch_X, batch_ret, batch_dir, batch_w in loader:
            optimizer.zero_grad()
            pred_ret, pred_dir_logits = model(batch_X)
            loss_return = ret_loss(pred_ret, batch_ret).mean(dim=1)
            loss_direction = dir_loss(pred_dir_logits, batch_dir).mean(dim=1)
            loss = (
                float(loss_weights.get("return", 1.0)) * loss_return
                + float(loss_weights.get("direction", 0.25)) * loss_direction
            )
            weighted = (loss * batch_w / (batch_w.mean() + 1e-8)).mean()
            weighted.backward()
            optimizer.step()
            total += float(weighted.detach().cpu())
            batches += 1
        losses.append(total / max(batches, 1))
    model.feature_mean = feature_mean.detach().cpu()
    model.feature_std = feature_std.detach().cpu()
    return model, {
        "training_losses": losses,
        "final_loss": losses[-1] if losses else 0.0,
        "preprocessing": {
            "scaler": "standard",
            "fit_scope": "train_indices_only",
            "mean_shape": list(feature_mean.shape),
            "std_shape": list(feature_std.shape),
        },
    }


def predict_base_model(model: MultiHorizonLSTM, dataset: Mapping[str, Any], indices: np.ndarray, device: torch.device, batch_size: int = 512) -> pd.DataFrame:
    X = dataset["X"][indices]
    pred_returns: List[torch.Tensor] = []
    pred_probs: List[torch.Tensor] = []
    model.eval()
    feature_mean = getattr(model, "feature_mean", torch.zeros((1, 1, int(dataset["input_size"]))))
    feature_std = getattr(model, "feature_std", torch.ones((1, 1, int(dataset["input_size"]))))
    feature_mean = feature_mean.to(device)
    feature_std = feature_std.to(device)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size].to(device)
            batch = (batch - feature_mean) / feature_std
            ret, logits = model(batch)
            pred_returns.append(ret.detach().cpu())
            pred_probs.append(torch.sigmoid(logits).detach().cpu())
    pred_ret = torch.cat(pred_returns, dim=0).numpy()
    pred_prob = torch.cat(pred_probs, dim=0).numpy()
    true_ret = dataset["y_ret"][indices].detach().cpu().numpy()
    true_dir = dataset["y_dir"][indices].detach().cpu().numpy()
    frame = dataset["samples"].iloc[indices].reset_index(drop=True).copy()
    for h_idx, horizon in enumerate(dataset["horizon_names"]):
        frame[f"{horizon}_pred_return"] = pred_ret[:, h_idx]
        frame[f"{horizon}_direction_prob"] = pred_prob[:, h_idx]
        frame[f"{horizon}_true_return"] = true_ret[:, h_idx]
        frame[f"{horizon}_true_direction"] = true_dir[:, h_idx]
    return frame


def _weighted_binary_classification_summary(y_true: Sequence[Any], y_prob: Sequence[Any], weights: Sequence[Any]) -> Dict[str, float]:
    truth = pd.to_numeric(pd.Series(y_true), errors="coerce").fillna(0.0).to_numpy(dtype=float) >= 0.5
    prob = pd.to_numeric(pd.Series(y_prob), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred = prob >= 0.5
    w = pd.to_numeric(pd.Series(weights), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    w = np.where(w > 0.0, w, 0.0)
    if len(w) != len(truth) or not np.any(w > 0.0):
        w = np.ones(len(truth), dtype=float)
    total = max(float(w.sum()), 1e-12)
    tp = float(w[truth & pred].sum())
    fp = float(w[~truth & pred].sum())
    fn = float(w[truth & ~pred].sum())
    accuracy = float(w[truth == pred].sum() / total)
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    epsilon = 1e-15
    clipped = np.clip(prob, epsilon, 1.0 - epsilon)
    log_loss = -float((w * (truth.astype(float) * np.log(clipped) + (1.0 - truth.astype(float)) * np.log(1.0 - clipped))).sum() / total)
    return {
        "weight_sum": total,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "log_loss": log_loss,
    }


def _weighted_average(values: Sequence[Any], weights: Sequence[Any], default: float = 0.0) -> float:
    clean_values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    clean_weights = pd.to_numeric(pd.Series(weights), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    mask = np.isfinite(clean_values) & np.isfinite(clean_weights) & (clean_weights > 0.0)
    if not np.any(mask):
        return float(default)
    return float(np.average(clean_values[mask], weights=clean_weights[mask]))


def _prediction_metrics(frame: pd.DataFrame, horizons: Sequence[str]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for horizon in horizons:
        metrics[horizon] = {
            "return": _sanitize_metrics(
                compute_metrics(
                    torch.tensor(frame[f"{horizon}_true_return"].to_numpy(dtype=float)).unsqueeze(1),
                    torch.tensor(frame[f"{horizon}_pred_return"].to_numpy(dtype=float)).unsqueeze(1),
                    "regression",
                )
            ),
            "direction": _sanitize_metrics(
                compute_metrics(
                    torch.tensor(frame[f"{horizon}_true_direction"].to_numpy(dtype=float)),
                    torch.tensor(frame[f"{horizon}_direction_prob"].to_numpy(dtype=float)),
                    "classification",
                )
            ),
        }
    if "sample_weight" in frame.columns:
        for horizon in horizons:
            metrics[horizon]["direction_weighted"] = _weighted_binary_classification_summary(
                frame[f"{horizon}_true_direction"],
                frame[f"{horizon}_direction_prob"],
                frame["sample_weight"],
            )
    return metrics


def candidate_signals(predictions: pd.DataFrame, dataset: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    label_cfg = config.get("labels", {}).get("meta_triple_barrier", {})
    min_edge_bps = float(label_cfg.get("min_edge_bps", 0.0))
    rows: List[Dict[str, Any]] = []
    horizon_names = list(dataset["horizon_names"])
    for _, row in predictions.iterrows():
        signs = [np.sign(float(row[f"{name}_pred_return"])) for name in horizon_names]
        disagreement = float(len(set(signs)) > 1) if len(signs) > 1 else 0.0
        shared = row.to_dict()
        for horizon in horizon_names:
            pred_return = float(row[f"{horizon}_pred_return"])
            side = int(np.sign(pred_return))
            direction_prob = float(row[f"{horizon}_direction_prob"])
            confidence = direction_prob if side >= 0 else 1.0 - direction_prob
            horizon_bars = int(dataset["horizon_bars"][horizon])
            cost_bps = _total_cost_bps(config, horizon_bars, str(dataset["granularity"]))
            expected_edge_bps = abs(pred_return) * 10_000.0 - cost_bps
            is_candidate = bool(side != 0 and expected_edge_bps >= min_edge_bps)
            out = dict(shared)
            out.update(
                {
                    "horizon": horizon,
                    "horizon_bars": horizon_bars,
                    "pred_return": pred_return,
                    "direction_prob": direction_prob,
                    "direction_confidence": confidence,
                    "true_return": float(row[f"{horizon}_true_return"]),
                    "true_direction": float(row[f"{horizon}_true_direction"]),
                    "side": side,
                    "total_cost_bps": cost_bps,
                    "expected_edge_bps": expected_edge_bps,
                    "is_candidate": is_candidate,
                    "horizon_disagreement": disagreement,
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def _candidate_weight_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([], index=frame.index, dtype=bool)
    if "is_candidate" not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)
    mask = frame["is_candidate"].fillna(False).astype(bool)
    if "meta_label" in frame.columns:
        mask = mask & frame["meta_label"].notna()
    return mask


def _apply_meta_sample_weights(candidates: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    out = candidates.copy()
    if "sample_weight" in out.columns and "base_sample_weight" not in out.columns:
        out["base_sample_weight"] = pd.to_numeric(out["sample_weight"], errors="coerce")
    out["sample_weight"] = 0.0
    weight_cfg = sample_weight_config(config)
    mask = _candidate_weight_mask(out)
    if bool(mask.any()):
        weights, diagnostics = sample_weights_and_diagnostics(
            out.loc[mask],
            mode=weight_cfg["meta_mode"],
            normalize=weight_cfg["normalize"],
        )
        out.loc[mask, "sample_weight"] = weights.to_numpy(dtype=float)
    else:
        diagnostics = sample_weight_diagnostics([], [], [], mode=weight_cfg["meta_mode"], normalize=weight_cfg["normalize"])
    out.attrs["sample_weight_diagnostics"] = {
        **diagnostics,
        "base_mode": weight_cfg["base_mode"],
        "meta_mode": weight_cfg["meta_mode"],
        "weighted_rows": int(mask.sum()),
        "total_rows": int(len(out)),
    }
    return out


def add_meta_labels(candidates: pd.DataFrame, dataset: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    labeled = meta_triple_barrier_labels(
        candidates,
        dataset["price_path"],
        config.get("labels", {}).get("meta_triple_barrier", {}),
        config.get("costs", {}),
        granularity=str(dataset["granularity"]),
    )
    return _apply_meta_sample_weights(labeled, config)


def _meta_feature_columns(frame: pd.DataFrame, horizons: Sequence[str]) -> List[str]:
    columns = [
        "pred_return",
        "direction_prob",
        "direction_confidence",
        "expected_edge_bps",
        "total_cost_bps",
        "realized_vol",
        "vol_regime",
        "horizon_bars",
        "side",
        "horizon_disagreement",
    ]
    for horizon in horizons:
        columns.extend([f"{horizon}_pred_return", f"{horizon}_direction_prob"])
    return [column for column in columns if column in frame.columns]


def fit_meta_labeler(candidates: pd.DataFrame, config: Mapping[str, Any], horizons: Sequence[str], device: torch.device) -> Any:
    feature_columns = _meta_feature_columns(candidates, horizons)
    train = candidates[(candidates["is_candidate"]) & candidates["meta_label"].notna()].copy()
    if train.empty:
        return ConstantMetaModel(0.0, feature_columns)
    y = train["meta_label"].astype(float).to_numpy()
    if "sample_weight" in train.columns:
        raw_weights = pd.to_numeric(train["sample_weight"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        raw_weights = np.where(raw_weights > 0.0, raw_weights, 0.0)
    else:
        raw_weights = np.ones(len(train), dtype=float)
    if not np.any(raw_weights > 0.0):
        raw_weights = np.ones(len(train), dtype=float)
    train_weights = raw_weights / max(float(raw_weights.mean()), 1e-12)
    if len(np.unique(y)) < 2:
        return ConstantMetaModel(float(np.average(y, weights=raw_weights)), feature_columns)
    X = train.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    meta_cfg = config.get("model", {}).get("meta_labeler", {})
    training = config.get("training", {})
    model = MetaLabelMLP(
        input_size=X.shape[1],
        hidden_size=int(meta_cfg.get("hidden_size", 16)),
        num_layers=int(meta_cfg.get("num_layers", 1)),
        dropout=float(meta_cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(training.get("learning_rate", 0.001)))
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(train_weights, dtype=torch.float32),
        ),
        batch_size=int(training.get("batch_size", 32)),
        shuffle=True,
    )
    for _ in range(int(training.get("epochs", 1))):
        model.train()
        for batch_X, batch_y, batch_w in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad()
            loss = (criterion(model(batch_X), batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()
    return TorchMetaModel(model, mean, std, feature_columns, device)


def add_meta_probabilities(candidates: pd.DataFrame, meta_model: Any) -> pd.DataFrame:
    out = candidates.copy()
    out["meta_prob"] = meta_model.predict_proba(out)
    out.loc[~out["is_candidate"], "meta_prob"] = 0.0
    return out


def _apply_alpha_trade_limits(
    eligible: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    initial_capital: float,
    notional: float,
) -> pd.DataFrame:
    alpha = config.get("alpha_backtest", {}) or {}
    out = eligible.copy()
    if out.empty:
        return out

    max_trades_per_day = alpha.get("max_trades_per_day")
    if max_trades_per_day is not None:
        out["_trade_day"] = pd.to_datetime(out["timestamp"]).dt.floor("D")
        out = out.groupby("_trade_day", sort=False).head(int(max_trades_per_day)).drop(columns=["_trade_day"])

    cooldown_bars = int(alpha.get("cooldown_bars", 0) or 0)
    if cooldown_bars > 0:
        gap = _granularity_timedelta(str(config.get("data", {}).get("granularity", DEFAULT_GRANULARITY)), bars=cooldown_bars)
        keep_indices: List[Any] = []
        last_timestamp: Optional[pd.Timestamp] = None
        for idx, row in out.sort_values("timestamp").iterrows():
            timestamp = pd.to_datetime(row["timestamp"])
            if last_timestamp is None or timestamp - last_timestamp >= gap:
                keep_indices.append(idx)
                last_timestamp = timestamp
        out = out.loc[keep_indices]

    max_turnover = alpha.get("max_turnover")
    if max_turnover is None and alpha.get("max_turnover_multiple") is not None:
        max_turnover = initial_capital * float(alpha.get("max_turnover_multiple"))
    if max_turnover is not None and notional > 0:
        max_trades = max(0, int(math.floor(float(max_turnover) / notional)))
        out = out.head(max_trades)

    return out


def _empty_alpha_metrics(candidate_count: int, total_candidates: int, policy: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "coverage": 0.0,
        "candidate_coverage": candidate_count / max(total_candidates, 1),
        "trades": 0.0,
        "gross_pnl": 0.0,
        "net_pnl": 0.0,
        "fees": 0.0,
        "gross_pnl_to_fees": 0.0,
        "turnover": 0.0,
        "exposure": 0.0,
        "average_net_pnl_per_trade": 0.0,
        "trades_per_day": 0.0,
        "hit_ratio": 0.0,
        "win_rate": 0.0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "max_drawdown": 0.0,
        "return_mean": 0.0,
        "return_std": 0.0,
        "return_p05": 0.0,
        "return_p50": 0.0,
        "return_p95": 0.0,
        "horizon_distribution": {},
        "side_distribution": {"long": 0, "short": 0},
        **dict(policy),
    }


def run_alpha_backtest(candidates: pd.DataFrame, config: Mapping[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    alpha = config.get("alpha_backtest", {}) or {}
    initial_capital = float(alpha.get("initial_capital", 100_000.0))
    meta_threshold = float(alpha.get("meta_threshold", 0.5))
    edge_threshold = float(alpha.get("edge_threshold_bps", 0.0))
    safety_margin = float(alpha.get("edge_safety_margin_bps", 0.0))
    effective_edge_threshold = edge_threshold + safety_margin
    horizon_choice = str(alpha.get("horizon", "best"))
    notional = float(alpha.get("position_notional", initial_capital))
    max_notional = float(alpha.get("max_position_notional", notional))
    notional = min(notional, max_notional) if max_notional > 0 else notional
    policy = {
        "meta_threshold": meta_threshold,
        "edge_threshold_bps": edge_threshold,
        "edge_safety_margin_bps": safety_margin,
        "effective_edge_threshold_bps": effective_edge_threshold,
        "max_trades_per_day": alpha.get("max_trades_per_day"),
        "cooldown_bars": int(alpha.get("cooldown_bars", 0) or 0),
    }

    eligible = candidates[
        (candidates["is_candidate"])
        & (candidates["meta_prob"] >= meta_threshold)
        & (candidates["expected_edge_bps"] >= effective_edge_threshold)
    ].copy()
    if horizon_choice != "best":
        eligible = eligible[eligible["horizon"] == horizon_choice].copy()
    if not eligible.empty:
        eligible["score"] = eligible["meta_prob"] * eligible["expected_edge_bps"]
        eligible = eligible.sort_values(["timestamp", "score"], ascending=[True, False])
        eligible = eligible.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp")
        eligible = _apply_alpha_trade_limits(eligible, config, initial_capital=initial_capital, notional=notional)

    trades: List[Dict[str, Any]] = []
    for _, row in eligible.iterrows():
        gross_value = row.get("label_gross_return", np.nan)
        if pd.isna(gross_value):
            gross_value = row["side"] * row["true_return"]
        gross_return = float(gross_value)
        net_value = row.get("label_net_return", np.nan)
        if pd.isna(net_value):
            net_value = gross_return - row["total_cost_bps"] / 10_000.0
        net_return = float(net_value)
        fee_cost = notional * float(row["total_cost_bps"]) / 10_000.0
        gross_pnl = notional * gross_return
        net_pnl = notional * net_return
        trades.append(
            {
                "timestamp": row["timestamp"],
                "horizon": row["horizon"],
                "side": int(row["side"]),
                "notional": notional,
                "entry_price": float(row["close"]),
                "gross_return": gross_return,
                "net_return": net_return,
                "gross_pnl": gross_pnl,
                "fees": fee_cost,
                "net_pnl": net_pnl,
                "meta_prob": float(row["meta_prob"]),
                "expected_edge_bps": float(row["expected_edge_bps"]),
                "exit_reason": row.get("exit_reason", "vertical"),
            }
        )
    trades_df = pd.DataFrame(trades)
    total_rows = max(1, int(candidates["timestamp"].nunique()) if "timestamp" in candidates else len(candidates))
    candidate_count = int(candidates["is_candidate"].sum()) if "is_candidate" in candidates else 0
    if trades_df.empty:
        return _empty_alpha_metrics(candidate_count, len(candidates), policy), pd.DataFrame(columns=["timestamp", "horizon", "side", "notional", "net_pnl"])

    trades_df["cumulative_net_pnl"] = trades_df["net_pnl"].cumsum()
    equity = initial_capital + trades_df["cumulative_net_pnl"]
    drawdown = equity - equity.cummax()
    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]
    returns = trades_df["net_return"].to_numpy(dtype=float)
    timestamps = pd.to_datetime(trades_df["timestamp"])
    days = max(1, int((timestamps.max().floor("D") - timestamps.min().floor("D")).days) + 1)
    horizon_distribution = {str(key): int(value) for key, value in trades_df["horizon"].value_counts().sort_index().items()}
    side_distribution = {
        "long": int((trades_df["side"] > 0).sum()),
        "short": int((trades_df["side"] < 0).sum()),
    }
    total_fees = float(trades_df["fees"].sum())
    gross_pnl = float(trades_df["gross_pnl"].sum())
    metrics = {
        "coverage": float(len(trades_df) / total_rows),
        "candidate_coverage": float(candidate_count / max(len(candidates), 1)),
        "trades": float(len(trades_df)),
        "gross_pnl": gross_pnl,
        "net_pnl": float(trades_df["net_pnl"].sum()),
        "fees": total_fees,
        "gross_pnl_to_fees": float(gross_pnl / total_fees) if abs(total_fees) > 1e-12 else 0.0,
        "turnover": float(trades_df["notional"].sum()),
        "exposure": float(trades_df["notional"].sum() / (initial_capital * total_rows)),
        "average_net_pnl_per_trade": float(trades_df["net_pnl"].mean()),
        "trades_per_day": float(len(trades_df) / days),
        "hit_ratio": float((trades_df["gross_return"] > 0).mean()),
        "win_rate": float((trades_df["net_pnl"] > 0).mean()),
        "average_win": float(wins["net_pnl"].mean()) if not wins.empty else 0.0,
        "average_loss": float(losses["net_pnl"].mean()) if not losses.empty else 0.0,
        "max_drawdown": float(abs(drawdown.min())) if len(drawdown) else 0.0,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_p05": float(np.quantile(returns, 0.05)),
        "return_p50": float(np.quantile(returns, 0.50)),
        "return_p95": float(np.quantile(returns, 0.95)),
        "horizon_distribution": horizon_distribution,
        "side_distribution": side_distribution,
        **policy,
    }
    return metrics, trades_df


def probability_bucket_table(candidates: pd.DataFrame, bins: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    bins = list(bins or np.linspace(0.0, 1.0, 11))
    frame = candidates[candidates["is_candidate"]].copy() if "is_candidate" in candidates else candidates.copy()
    if frame.empty or "meta_prob" not in frame:
        return []
    frame["bucket"] = pd.cut(frame["meta_prob"].astype(float), bins=bins, include_lowest=True)
    rows: List[Dict[str, Any]] = []
    for bucket, part in frame.groupby("bucket", observed=False):
        labeled = part[part["meta_label"].notna()] if "meta_label" in part else part.iloc[0:0]
        rows.append(
            {
                "bucket": str(bucket),
                "count": int(len(part)),
                "mean_meta_prob": float(part["meta_prob"].mean()) if len(part) else 0.0,
                "observed_success_rate": float(labeled["meta_label"].mean()) if len(labeled) else 0.0,
                "weighted_observed_success_rate": _weighted_average(labeled["meta_label"], labeled["sample_weight"]) if "sample_weight" in labeled and len(labeled) else 0.0,
                "mean_expected_edge_bps": float(part["expected_edge_bps"].mean()) if "expected_edge_bps" in part and len(part) else 0.0,
            }
        )
    return rows


def predicted_edge_bucket_table(candidates: pd.DataFrame, bins: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    bins = list(bins or [-math.inf, 0.0, 5.0, 10.0, 25.0, 50.0, math.inf])
    labels = ["lt_0", "0_5", "5_10", "10_25", "25_50", "50_plus"]
    if candidates.empty or "expected_edge_bps" not in candidates:
        return []
    frame = candidates[candidates["is_candidate"]].copy() if "is_candidate" in candidates else candidates.copy()
    frame["bucket"] = pd.cut(frame["expected_edge_bps"].astype(float), bins=bins, labels=labels, include_lowest=True)
    rows: List[Dict[str, Any]] = []
    for label in labels:
        part = frame[frame["bucket"] == label]
        labeled = part[part["meta_label"].notna()] if "meta_label" in part else part.iloc[0:0]
        rows.append(
            {
                "bucket": label,
                "count": int(len(part)),
                "mean_expected_edge_bps": float(part["expected_edge_bps"].mean()) if len(part) else 0.0,
                "observed_success_rate": float(labeled["meta_label"].mean()) if len(labeled) else 0.0,
                "weighted_observed_success_rate": _weighted_average(labeled["meta_label"], labeled["sample_weight"]) if "sample_weight" in labeled and len(labeled) else 0.0,
                "mean_label_net_return": float(part["label_net_return"].mean()) if "label_net_return" in part and len(part) else 0.0,
            }
        )
    return rows


def meta_threshold_table(candidates: pd.DataFrame, thresholds: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    thresholds = list(thresholds or [0.50, 0.55, 0.60, 0.65, 0.70, 0.80])
    if candidates.empty or "meta_prob" not in candidates or "meta_label" not in candidates:
        return []
    base = candidates[candidates["is_candidate"]].copy() if "is_candidate" in candidates else candidates.copy()
    labeled = base[base["meta_label"].notna()].copy()
    positives = float((labeled["meta_label"] == 1.0).sum())
    if "sample_weight" in labeled.columns:
        label_weights = pd.to_numeric(labeled["sample_weight"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        label_weights = np.where(label_weights > 0.0, label_weights, 0.0)
        if not np.any(label_weights > 0.0):
            label_weights = np.ones(len(labeled), dtype=float)
    else:
        label_weights = np.ones(len(labeled), dtype=float)
    labeled = labeled.copy()
    labeled["_threshold_weight"] = label_weights
    positive_weight = float(labeled.loc[labeled["meta_label"] == 1.0, "_threshold_weight"].sum())
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        selected = labeled[labeled["meta_prob"] >= float(threshold)]
        tp = float((selected["meta_label"] == 1.0).sum())
        selected_weight = float(selected["_threshold_weight"].sum()) if len(selected) else 0.0
        selected_positive_weight = float(selected.loc[selected["meta_label"] == 1.0, "_threshold_weight"].sum()) if len(selected) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "selected": int(len(selected)),
                "coverage": float(len(selected) / max(len(labeled), 1)),
                "precision": float(tp / max(len(selected), 1)),
                "recall": float(tp / max(positives, 1.0)),
                "weighted_precision": float(selected_positive_weight / max(selected_weight, 1e-12)),
                "weighted_recall": float(selected_positive_weight / max(positive_weight, 1e-12)),
            }
        )
    return rows


def alpha_diagnostics(candidates: pd.DataFrame, config: Mapping[str, Any]) -> Dict[str, Any]:
    alpha = config.get("alpha_backtest", {}) or {}
    thresholds = alpha.get("meta_threshold_grid")
    return {
        "probability_buckets": probability_bucket_table(candidates),
        "predicted_edge_buckets": predicted_edge_bucket_table(candidates),
        "meta_label_thresholds": meta_threshold_table(candidates, thresholds),
    }


def _split_manifest(dataset: Mapping[str, Any], splits: Mapping[str, Any]) -> Dict[str, Any]:
    samples = dataset["samples"]

    def describe(indices: np.ndarray) -> Dict[str, Any]:
        if len(indices) == 0:
            return {"samples": 0}
        part = samples.iloc[indices]
        return {
            "samples": int(len(indices)),
            "start_index": int(indices[0]),
            "end_index": int(indices[-1]),
            "start_timestamp": part["timestamp"].iloc[0],
            "end_timestamp": part["timestamp"].iloc[-1],
        }

    return {
        "method": "purged_walk_forward",
        "purge_bars": int(splits["purge_bars"]),
        "embargo_bars": int(splits["embargo_bars"]),
        "development": describe(splits["development"]),
        "gap": describe(splits["gap"]),
        "test": describe(splits["test"]),
        "folds": [
            {"fold": int(fold["fold"]), "train": describe(fold["train"]), "validation": describe(fold["validation"])}
            for fold in splits["folds"]
        ],
    }


def _write_stage0(dataset: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    stage_dir = run_dir / "stage0_features"
    stage_dir.mkdir(parents=True, exist_ok=True)
    samples_path = stage_dir / "samples.parquet"
    dataset["samples"].to_parquet(samples_path, index=False)
    manifest = {
        "stage": "stage0_features",
        "path": stage_dir,
        "target": "multi_horizon",
        "granularity": dataset["granularity"],
        "sequence_length": dataset["sequence_length"],
        "horizons": dataset["horizon_bars"],
        "feature_columns": dataset["feature_columns"],
        "feature_window_bars": dataset["feature_window_bars"],
        "feature_manifest": dataset.get("feature_manifest", {}),
        "bar_manifest": dataset.get("bar_manifest", {}),
        "event_diagnostics": dataset.get("event_diagnostics", {}),
        "sample_weight_diagnostics": dataset.get("sample_weight_diagnostics", {}),
        "bar_clock_diagnostics": dataset.get("bar_clock_diagnostics", {}),
        "side_data_manifest": dataset.get("side_data_manifest", {}),
        "input_size": int(dataset["input_size"]),
        "samples": int(dataset["X"].shape[0]),
        "X_shape": list(dataset["X"].shape),
        "y_ret_shape": list(dataset["y_ret"].shape),
        "samples_path": samples_path,
    }
    _write_json(stage_dir / "manifest.json", manifest)
    return manifest


def _run_one_trial(spec: Mapping[str, Any], run_dir: Path, *, smoke: bool, device: torch.device) -> Dict[str, Any]:
    config = dict(spec["config"])
    if smoke:
        config = apply_smoke_overrides(config)
    trial_id = str(spec["trial_id"])
    trial_dir = run_dir / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config.get("experiment", {}).get("seed", 42)) + int(spec["trial_index"]))
    np.random.seed(int(config.get("experiment", {}).get("seed", 42)) + int(spec["trial_index"]))

    dataset = build_multi_horizon_lighter_dataset(config, smoke=smoke)
    splits = purged_walk_forward_splits(int(dataset["X"].shape[0]), config.get("validation", {}))
    split_manifest = _split_manifest(dataset, splits)
    split_manifest_path = trial_dir / "split_manifest.json"
    _write_json(split_manifest_path, split_manifest)
    split_manifest_hash = _stable_hash(split_manifest)

    oof_frames: List[pd.DataFrame] = []
    fold_manifests: List[Dict[str, Any]] = []
    batch_size = int(config.get("training", {}).get("prediction_batch_size", 512))
    for fold in splits["folds"]:
        model, history = train_base_model(dataset, fold["train"], config, device)
        fold_pred = predict_base_model(model, dataset, fold["validation"], device, batch_size=batch_size)
        fold_pred["fold"] = int(fold["fold"])
        oof_frames.append(fold_pred)
        fold_manifests.append(
            {
                "fold": int(fold["fold"]),
                "train_samples": int(len(fold["train"])),
                "validation_samples": int(len(fold["validation"])),
                "history": history,
                "metrics": _prediction_metrics(fold_pred, dataset["horizon_names"]),
            }
        )

    oof_predictions = pd.concat(oof_frames, ignore_index=True).sort_values("timestamp")
    final_model, final_history = train_base_model(dataset, splits["development"], config, device)
    test_predictions = predict_base_model(final_model, dataset, splits["test"], device, batch_size=batch_size)

    oof_candidates = add_meta_labels(candidate_signals(oof_predictions, dataset, config), dataset, config)
    test_candidates = add_meta_labels(candidate_signals(test_predictions, dataset, config), dataset, config)
    meta_sample_weight_diagnostics = {
        "validation": oof_candidates.attrs.get("sample_weight_diagnostics", {}),
        "test": test_candidates.attrs.get("sample_weight_diagnostics", {}),
    }
    meta_model = fit_meta_labeler(oof_candidates, config, dataset["horizon_names"], device)
    oof_candidates = add_meta_probabilities(oof_candidates, meta_model)
    test_candidates = add_meta_probabilities(test_candidates, meta_model)

    validation_metrics, validation_trades = run_alpha_backtest(oof_candidates, config)
    test_metrics, test_trades = run_alpha_backtest(test_candidates, config)
    diagnostics = {"validation": alpha_diagnostics(oof_candidates, config), "test": alpha_diagnostics(test_candidates, config)}
    final_test_evaluation = _record_final_test_evaluation(config, run_dir, trial_id=trial_id, split_hash=split_manifest_hash, smoke=smoke)

    model_path = trial_dir / "base_multi_horizon_lstm.pt"
    torch.save(final_model.state_dict(), model_path)
    paths = {
        "oof_predictions": trial_dir / "predictions_oof.parquet",
        "test_predictions": trial_dir / "predictions_test.parquet",
        "oof_candidates": trial_dir / "meta_candidates_oof.parquet",
        "test_candidates": trial_dir / "meta_candidates_test.parquet",
        "validation_trades": trial_dir / "alpha_trades_validation.parquet",
        "test_trades": trial_dir / "alpha_trades_test.parquet",
        "diagnostics": trial_dir / "alpha_diagnostics.json",
    }
    oof_predictions.to_parquet(paths["oof_predictions"], index=False)
    test_predictions.to_parquet(paths["test_predictions"], index=False)
    oof_candidates.to_parquet(paths["oof_candidates"], index=False)
    test_candidates.to_parquet(paths["test_candidates"], index=False)
    validation_trades.to_parquet(paths["validation_trades"], index=False)
    test_trades.to_parquet(paths["test_trades"], index=False)
    _write_json(paths["diagnostics"], diagnostics)

    manifest = {
        "stage": "mvp_trial",
        "trial_index": int(spec["trial_index"]),
        "trial_id": trial_id,
        "status": "completed",
        "overrides": spec.get("overrides", {}),
        "config": config,
        "dataset": {
            "samples": int(dataset["X"].shape[0]),
            "input_size": int(dataset["input_size"]),
            "sequence_length": int(dataset["sequence_length"]),
            "horizons": dataset["horizon_bars"],
            "feature_manifest": dataset.get("feature_manifest", {}),
            "bar_manifest": dataset.get("bar_manifest", {}),
            "event_diagnostics": dataset.get("event_diagnostics", {}),
            "sample_weight_diagnostics": dataset.get("sample_weight_diagnostics", {}),
            "bar_clock_diagnostics": dataset.get("bar_clock_diagnostics", {}),
            "side_data_manifest": dataset.get("side_data_manifest", {}),
        },
        "run_identity": {"config_hash": _stable_hash(config), "git": _git_manifest(), "dependencies": _dependency_manifest()},
        "raw_data": _raw_data_manifest(config),
        "split_manifest_path": split_manifest_path,
        "split_manifest_hash": split_manifest_hash,
        "final_test_evaluation": final_test_evaluation,
        "folds": fold_manifests,
        "final_training_history": final_history,
        "meta_labeler": meta_model.manifest(),
        "meta_sample_weight_diagnostics": meta_sample_weight_diagnostics,
        "model_path": model_path,
        "artifact_paths": paths,
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "diagnostics": diagnostics,
        "diagnostics_path": paths["diagnostics"],
        "manifest_path": trial_dir / "manifest.json",
    }
    _write_json(trial_dir / "manifest.json", manifest)
    return manifest


def _failed_trial_manifest(spec: Mapping[str, Any], run_dir: Path, *, smoke: bool, error: BaseException) -> Dict[str, Any]:
    config = dict(spec.get("config", {}))
    if smoke:
        config = apply_smoke_overrides(config)
    trial_id = str(spec.get("trial_id", "unknown"))
    trial_dir = run_dir / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "mvp_trial",
        "trial_index": int(spec.get("trial_index", -1)),
        "trial_id": trial_id,
        "status": "failed",
        "reason": str(error),
        "skip_reason": str(error),
        "error_type": type(error).__name__,
        "traceback": traceback.format_exc(),
        "overrides": spec.get("overrides", {}),
        "config": config,
        "run_identity": {"config_hash": _stable_hash(config), "git": _git_manifest(), "dependencies": _dependency_manifest()},
        "raw_data": _raw_data_manifest(config),
        "artifact_paths": {},
        "metrics": {"validation": {}, "test": {}},
        "manifest_path": trial_dir / "manifest.json",
    }
    _write_json(trial_dir / "manifest.json", manifest)
    return manifest


def _skipped_trial_manifest(spec: Mapping[str, Any], run_dir: Path, *, smoke: bool, reason: str) -> Dict[str, Any]:
    config = dict(spec.get("config", {}))
    if smoke:
        config = apply_smoke_overrides(config)
    trial_id = str(spec.get("trial_id", "unknown"))
    trial_dir = run_dir / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "mvp_trial",
        "trial_index": int(spec.get("trial_index", -1)),
        "trial_id": trial_id,
        "status": "skipped",
        "reason": reason,
        "skip_reason": reason,
        "overrides": spec.get("overrides", {}),
        "config": config,
        "run_identity": {"config_hash": _stable_hash(config), "git": _git_manifest(), "dependencies": _dependency_manifest()},
        "raw_data": _raw_data_manifest(config),
        "artifact_paths": {},
        "metrics": {"validation": {}, "test": {}},
        "manifest_path": trial_dir / "manifest.json",
    }
    _write_json(trial_dir / "manifest.json", manifest)
    return manifest


def _run_trial_safe(spec: Mapping[str, Any], run_dir: Path, *, smoke: bool, device: torch.device) -> Dict[str, Any]:
    trial_cfg = (spec.get("config", {}) or {}).get("trial", {})
    if isinstance(trial_cfg, Mapping) and bool(trial_cfg.get("skip", False)):
        return _skipped_trial_manifest(
            spec,
            run_dir,
            smoke=smoke,
            reason=str(trial_cfg.get("skip_reason", "trial marked skipped by config")),
        )
    try:
        return _run_one_trial(spec, run_dir, smoke=smoke, device=device)
    except RuntimeError as exc:
        if "Final test set reuse blocked" in str(exc):
            raise
        return _failed_trial_manifest(spec, run_dir, smoke=smoke, error=exc)
    except Exception as exc:
        return _failed_trial_manifest(spec, run_dir, smoke=smoke, error=exc)


def _metric_value(payload: Mapping[str, Any], path: str) -> float:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return -math.inf
        current = current[part]
    return _safe_float(current, -math.inf)


def _rank_trials(trials: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    pipeline = config.get("pipeline", {}) or {}
    metric = str(pipeline.get("selection_metric", "metrics.validation.net_pnl"))
    mode = str(pipeline.get("selection_mode", "max"))
    return sorted(trials, key=lambda item: _metric_value(item, metric), reverse=(mode == "max"))


def _validation_trade_count(trial: Mapping[str, Any]) -> float:
    return _metric_value(trial, "metrics.validation.trades")


def _mean_validation_direction_accuracy(trial: Mapping[str, Any]) -> float:
    values: List[float] = []
    for fold in trial.get("folds", []) or []:
        fold_metrics = fold.get("metrics", {}) if isinstance(fold, Mapping) else {}
        for horizon_metrics in (fold_metrics or {}).values():
            if not isinstance(horizon_metrics, Mapping):
                continue
            direction = horizon_metrics.get("direction", {})
            value = _safe_float(direction.get("accuracy"), math.nan) if isinstance(direction, Mapping) else math.nan
            if math.isfinite(value):
                values.append(value)
    return float(np.mean(values)) if values else -math.inf


def _trial_calibration_error(trial: Mapping[str, Any]) -> float:
    diagnostics = trial.get("diagnostics", {}) if isinstance(trial.get("diagnostics"), Mapping) else {}
    validation = diagnostics.get("validation", {}) if isinstance(diagnostics, Mapping) else {}
    if not validation:
        return math.inf
    return _expected_calibration_error(validation)


def _selection_roles_for_trial(trial: Mapping[str, Any], selection: Mapping[str, Any]) -> List[str]:
    trial_id = trial.get("trial_id")
    roles: List[str] = []
    if trial_id == selection.get("raw_best_trial_id"):
        roles.append("raw_best")
    if trial_id == selection.get("best_trading_trial_id"):
        roles.append("trade_qualified_best")
    if trial_id == selection.get("classification_best_trial_id"):
        roles.append("classification_best")
    if trial_id == selection.get("calibration_best_trial_id"):
        roles.append("calibration_best")
    return roles or ["not_selected"]


def select_trials_with_trade_floor(
    trials: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    pipeline = config.get("pipeline", {}) or {}
    metric = str(pipeline.get("selection_metric", "metrics.validation.net_pnl"))
    mode = str(pipeline.get("selection_mode", "max"))
    min_trades = int(pipeline.get("min_validation_trades", pipeline.get("selection_min_trades", 1)) or 0)
    ranked = _rank_trials(trials, config)
    raw_best = dict(ranked[0])
    raw_best_trades = _validation_trade_count(raw_best)
    raw_best["selected_by"] = {"metric": metric, "mode": mode, "value": _metric_value(raw_best, metric)}
    raw_best["selection_status"] = "trade_qualified" if raw_best_trades >= min_trades else "abstention"

    qualified = [trial for trial in trials if _validation_trade_count(trial) >= min_trades]
    best_trading: Optional[Dict[str, Any]] = None
    if qualified:
        best_trading = dict(_rank_trials(qualified, config)[0])
        best_trading["selected_by"] = {"metric": metric, "mode": mode, "value": _metric_value(best_trading, metric)}
        best_trading["selection_status"] = "trade_qualified"

    classification_best = max(trials, key=_mean_validation_direction_accuracy)
    calibration_candidates = [trial for trial in trials if math.isfinite(_trial_calibration_error(trial))]
    calibration_best = min(calibration_candidates, key=_trial_calibration_error) if calibration_candidates else None

    selection = {
        "metric": metric,
        "mode": mode,
        "min_validation_trades": min_trades,
        "raw_best_trial_id": raw_best.get("trial_id"),
        "raw_best_validation_trades": raw_best_trades,
        "raw_best_selection_status": raw_best["selection_status"],
        "best_trading_trial_id": best_trading.get("trial_id") if best_trading else None,
        "classification_best_trial_id": classification_best.get("trial_id") if classification_best else None,
        "classification_best_score": _mean_validation_direction_accuracy(classification_best) if classification_best else None,
        "calibration_best_trial_id": calibration_best.get("trial_id") if calibration_best else None,
        "calibration_best_ece": _trial_calibration_error(calibration_best) if calibration_best else None,
        "qualified_trial_count": int(len(qualified)),
    }
    return raw_best, best_trading, selection


def _expected_calibration_error(diagnostics: Mapping[str, Any]) -> float:
    buckets = diagnostics.get("probability_buckets", []) if isinstance(diagnostics, Mapping) else []
    total = sum(int(row.get("count", 0) or 0) for row in buckets)
    if total <= 0:
        return 0.0
    error = 0.0
    for row in buckets:
        count = int(row.get("count", 0) or 0)
        error += count * abs(_safe_float(row.get("mean_meta_prob")) - _safe_float(row.get("observed_success_rate")))
    return float(error / total)


def _trial_failure_modes(
    trial: Mapping[str, Any],
    *,
    selection_role: str,
    min_trades: int,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    thresholds = ((config.get("tracking", {}) or {}).get("failure_mode_thresholds", {}) or {})
    initial_capital = _safe_float((config.get("alpha_backtest", {}) or {}).get("initial_capital"), 1.0)
    validation = trial.get("metrics", {}).get("validation", {}) if isinstance(trial.get("metrics"), Mapping) else {}
    test = trial.get("metrics", {}).get("test", {}) if isinstance(trial.get("metrics"), Mapping) else {}
    diagnostics = trial.get("diagnostics", {}) if isinstance(trial.get("diagnostics"), Mapping) else {}
    validation_trades = _safe_float(validation.get("trades"))
    test_trades = _safe_float(test.get("trades"))
    validation_turnover = _safe_float(validation.get("turnover"))
    validation_fees = _safe_float(validation.get("fees"))
    validation_gross = _safe_float(validation.get("gross_pnl"))
    validation_net = _safe_float(validation.get("net_pnl"))
    test_net = _safe_float(test.get("net_pnl"))
    calibration_error = _expected_calibration_error(diagnostics.get("validation", {}))
    is_winner = selection_role in {"raw_best", "trade_qualified_best", "classification_best", "calibration_best", "raw_best_trade_qualified", "trade_qualified_selected", "abstention", "trade_qualified"}
    cost_ratio = abs(validation_fees) / max(abs(validation_gross), 1e-8)
    flags = {
        "no_trade_winner": bool(is_winner and validation_trades <= 0),
        "low_trade_winner": bool(is_winner and validation_trades < min_trades),
        "poor_calibration": bool(calibration_error > float(thresholds.get("poor_calibration_ece", 0.20))),
        "high_turnover": bool(validation_turnover > initial_capital * float(thresholds.get("high_turnover_multiple", 10.0))),
        "unstable_validation_test": bool((validation_net > 0.0 and test_net < 0.0) or (validation_net < 0.0 and test_net > 0.0)),
        "high_cost_sensitivity": bool(cost_ratio > float(thresholds.get("high_cost_to_gross_ratio", 2.0))),
        "concentrated_pnl": bool(validation_trades > 0 and _safe_float(validation.get("coverage")) < float(thresholds.get("min_trade_coverage", 0.01))),
    }
    return {
        "flags": flags,
        "metrics": {
            "validation_trades": validation_trades,
            "test_trades": test_trades,
            "validation_turnover": validation_turnover,
            "validation_net_pnl": validation_net,
            "test_net_pnl": test_net,
            "calibration_error": calibration_error,
            "cost_to_gross_ratio": cost_ratio,
        },
    }


def _trial_accounting(trials: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    min_trades = int(selection.get("min_validation_trades", 0) or 0)
    rows: List[Dict[str, Any]] = []
    counts = {"completed": 0, "failed": 0, "skipped": 0, "no_trade": 0, "low_trade": 0, "trade_qualified": 0}
    flag_counts = {
        "no_trade_winner": 0,
        "low_trade_winner": 0,
        "poor_calibration": 0,
        "high_turnover": 0,
        "unstable_validation_test": 0,
        "high_cost_sensitivity": 0,
        "concentrated_pnl": 0,
    }
    for trial in trials:
        validation_trades = _metric_value(trial, "metrics.validation.trades")
        test_trades = _metric_value(trial, "metrics.test.trades")
        validation_trades = validation_trades if math.isfinite(validation_trades) else 0.0
        test_trades = test_trades if math.isfinite(test_trades) else 0.0
        status = str(trial.get("status", "completed"))
        trial_id = trial.get("trial_id")
        selection_roles = _selection_roles_for_trial(trial, selection)
        selection_role = selection_roles[0]
        failure_modes = _trial_failure_modes(trial, selection_role=selection_role, min_trades=min_trades, config=config)
        for flag, value in failure_modes["flags"].items():
            flag_counts[flag] = flag_counts.get(flag, 0) + int(bool(value))
        counts[status] = counts.get(status, 0) + 1
        if status == "completed":
            if validation_trades <= 0:
                counts["no_trade"] += 1
            if validation_trades < min_trades:
                counts["low_trade"] += 1
            else:
                counts["trade_qualified"] += 1
        rows.append(
            {
                "trial_id": trial_id,
                "status": status,
                "validation_trades": validation_trades,
                "test_trades": test_trades,
                "selection_role": selection_role,
                "selection_roles": selection_roles,
                "skip_reason": trial.get("skip_reason"),
                "artifact_paths": trial.get("artifact_paths", {}),
                "split_manifest_hash": trial.get("split_manifest_hash"),
                "failure_modes": failure_modes,
            }
        )
    return {
        "trial_count": int(len(trials)),
        "counts": counts,
        "trials": rows,
        "failure_modes": {
            "no_trade_trials": counts.get("no_trade", 0),
            "low_trade_trials": counts.get("low_trade", 0),
            "qualified_trial_count": counts.get("trade_qualified", 0),
            **flag_counts,
        },
    }


def _selection_status_for_trial(
    trial: Mapping[str, Any],
    raw_best: Mapping[str, Any],
    best_trading: Optional[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> str:
    trial_id = trial.get("trial_id")
    raw_best_id = raw_best.get("trial_id")
    best_trading_id = best_trading.get("trial_id") if best_trading else None
    if trial_id == raw_best_id and trial_id == best_trading_id:
        return "raw_best_trade_qualified"
    if trial_id == raw_best_id:
        return str(raw_best.get("selection_status", "raw_best"))
    if trial_id == best_trading_id:
        return "trade_qualified_selected"
    roles = _selection_roles_for_trial(trial, selection)
    if "classification_best" in roles and "calibration_best" in roles:
        return "classification_calibration_best"
    if "classification_best" in roles:
        return "classification_best"
    if "calibration_best" in roles:
        return "calibration_best"
    min_trades = int(selection.get("min_validation_trades", 0) or 0)
    if _validation_trade_count(trial) >= min_trades:
        return "trade_qualified_not_selected"
    return "unselected"


def _log_meta_label_trial_trackio(
    config: Mapping[str, Any],
    *,
    run_id: str,
    trial: Mapping[str, Any],
    selection_status: str,
    raw_best: Mapping[str, Any],
    best_trading: Optional[Mapping[str, Any]],
    selection: Mapping[str, Any],
    smoke: bool,
) -> None:
    trial_id = str(trial.get("trial_id", "unknown"))
    artifacts = {"manifest": trial.get("manifest_path"), "diagnostics": trial.get("diagnostics_path"), "model": trial.get("model_path")}
    artifacts.update(trial.get("artifact_paths", {}) or {})
    artifacts = {key: value for key, value in artifacts.items() if value is not None}
    is_raw_best = trial_id == raw_best.get("trial_id")
    is_best_trading = bool(best_trading and trial_id == best_trading.get("trial_id"))
    selection_roles = _selection_roles_for_trial(trial, selection)

    run_config = {
        "stage": "meta_label_mvp_trial",
        "run_id": run_id,
        "trial_id": trial_id,
        "trial_index": trial.get("trial_index"),
        "selection_status": selection_status,
        "selection_roles": selection_roles,
        "is_raw_best": is_raw_best,
        "is_best_trade_qualified": is_best_trading,
        "smoke": smoke,
        "overrides": trial.get("overrides", {}),
        "manifest_path": trial.get("manifest_path"),
        "diagnostics_path": trial.get("diagnostics_path"),
        "model_path": trial.get("model_path"),
        "trial_status": trial.get("status", "completed"),
        "split_manifest_hash": trial.get("split_manifest_hash"),
        "final_test_evaluation": trial.get("final_test_evaluation"),
        "feature_manifest": (trial.get("dataset", {}) or {}).get("feature_manifest"),
        "bar_manifest": (trial.get("dataset", {}) or {}).get("bar_manifest"),
        "event_diagnostics": (trial.get("dataset", {}) or {}).get("event_diagnostics"),
        "sample_weight_diagnostics": (trial.get("dataset", {}) or {}).get("sample_weight_diagnostics"),
        "bar_clock_diagnostics": (trial.get("dataset", {}) or {}).get("bar_clock_diagnostics"),
        "side_data_manifest": (trial.get("dataset", {}) or {}).get("side_data_manifest"),
    }
    metrics = {
        "alpha": trial.get("metrics", {}) or {},
        "selection": {
            "is_raw_best": is_raw_best,
            "is_best_trade_qualified": is_best_trading,
            "is_classification_best": "classification_best" in selection_roles,
            "is_calibration_best": "calibration_best" in selection_roles,
            "trade_qualified": selection_status in {"raw_best_trade_qualified", "trade_qualified_selected", "trade_qualified_not_selected"},
        },
        "trial_accounting": {
            "final_test_reuse_count": (trial.get("final_test_evaluation", {}) or {}).get("count", 0),
            "validation_trades": _metric_value(trial, "metrics.validation.trades"),
            "test_trades": _metric_value(trial, "metrics.test.trades"),
            "failure_modes": _trial_failure_modes(
                trial,
                selection_role=selection_status,
                min_trades=int((config.get("pipeline", {}) or {}).get("min_validation_trades", 1) or 0),
                config=config,
            ),
        },
    }
    log_trackio_run(
        config,
        name=f"meta_label/{run_id}/{trial_id}",
        group="meta_label_mvp",
        run_config=run_config,
        metrics=metrics,
        artifacts=artifacts,
        status=str(trial.get("status", "completed")),
    )


def _log_meta_label_summary_trackio(config: Mapping[str, Any], *, run_id: str, report: Mapping[str, Any], smoke: bool) -> None:
    best = report.get("best_trial", {}) or {}
    best_trading = report.get("best_trading_trial") or {}
    selection = report.get("selection", {}) or {}
    benchmark = report.get("benchmark") or {}
    report_paths = {
        "json": report.get("report_json_path"),
        "markdown": report.get("report_markdown_path"),
        "manifest": Path(report.get("run_dir", ".")) / "manifest.json" if report.get("run_dir") else None,
        "raw_best_manifest": report.get("best_trial_manifest_path"),
        "trade_qualified_manifest": report.get("best_trading_trial_manifest_path"),
        "benchmark_manifest": benchmark.get("manifest_path") if isinstance(benchmark, Mapping) else None,
    }
    artifacts = {key: value for key, value in report_paths.items() if value is not None}
    run_config = {
        "stage": "meta_label_mvp_summary",
        "run_id": run_id,
        "smoke": smoke,
        "config_path": report.get("config_path"),
        "run_dir": report.get("run_dir"),
        "raw_best_trial_id": selection.get("raw_best_trial_id"),
        "raw_best_selection_status": selection.get("raw_best_selection_status"),
        "best_trading_trial_id": selection.get("best_trading_trial_id"),
        "qualified_trial_count": selection.get("qualified_trial_count"),
        "report_paths": report_paths,
        "run_identity": report.get("run_identity"),
        "trial_accounting_path": report.get("trial_accounting_path"),
    }
    metrics = {
        "selection": selection,
        "raw_best": best.get("metrics", {}) or {},
        "best_trading": best_trading.get("metrics", {}) if isinstance(best_trading, Mapping) else {},
        "run": {
            "trials": len(report.get("trials", []) or []),
            "smoke": smoke,
            "has_trade_qualified_trial": bool(report.get("best_trading_trial")),
        },
        "trial_accounting": report.get("trial_accounting", {}) or {},
    }
    log_trackio_run(
        config,
        name=f"meta_label/{run_id}/summary",
        group="meta_label_mvp",
        run_config=run_config,
        metrics=metrics,
        artifacts=artifacts,
        status=str(report.get("status", "success")),
    )


def _report_markdown(report: Mapping[str, Any]) -> str:
    best = report["best_trial"]
    best_trading = report.get("best_trading_trial")
    selection = report.get("selection", {})
    lines = [
        "# Meta-Labeling MVP Report: {}".format(report["run_id"]),
        "",
        "- Status: `{}`".format(report["status"]),
        "- Trials: `{}`".format(len(report["trials"])),
        "- Raw best trial: `{}`".format(best["trial_id"]),
        "- Raw best status: `{}`".format(best.get("selection_status", "unknown")),
        "- Raw validation net PnL: `{}`".format(best["metrics"]["validation"]["net_pnl"]),
        "- Raw validation trades: `{}`".format(best["metrics"]["validation"]["trades"]),
        "- Test net PnL: `{}`".format(best["metrics"]["test"]["net_pnl"]),
        "- Test trades: `{}`".format(best["metrics"]["test"]["trades"]),
        "- Test coverage: `{}`".format(best["metrics"]["test"]["coverage"]),
        "- Minimum validation trades for trading selection: `{}`".format(selection.get("min_validation_trades")),
    ]
    if best_trading:
        lines.extend(
            [
                "- Best trade-qualified trial: `{}`".format(best_trading["trial_id"]),
                "- Trade-qualified validation net PnL: `{}`".format(best_trading["metrics"]["validation"]["net_pnl"]),
                "- Trade-qualified validation trades: `{}`".format(best_trading["metrics"]["validation"]["trades"]),
            ]
        )
    else:
        lines.append("- Best trade-qualified trial: `none`")
    if report.get("benchmark"):
        lines.append("- Forecast benchmark: `{}`".format(report["benchmark"]["manifest_path"]))
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Report JSON: `{}`".format(report["report_json_path"]),
            "- Raw best trial manifest: `{}`".format(best["manifest_path"]),
        ]
    )
    if report.get("best_trading_trial_manifest_path"):
        lines.append("- Trade-qualified trial manifest: `{}`".format(report["best_trading_trial_manifest_path"]))
    return "\n".join(lines) + "\n"


def run_meta_labeling_mvp(
    config_path: Path,
    *,
    smoke: bool = False,
    device: str = "cuda:0",
    allow_cpu: bool = False,
    run_name: Optional[str] = None,
    artifact_root: Optional[Path] = None,
) -> Dict[str, Any]:
    base_config = _load_yaml(config_path)
    config_for_search = apply_smoke_overrides(base_config) if smoke else deepcopy(base_config)
    if run_name:
        config_for_search.setdefault("pipeline", {})["run_name"] = run_name
    if artifact_root:
        config_for_search.setdefault("pipeline", {})["artifact_root"] = str(artifact_root)

    enforce_trackio_policy(config_for_search, smoke=smoke)

    pipeline = config_for_search.get("pipeline", {}) or {}
    run_id = _run_id(str(pipeline.get("run_name", "meta_label_mvp")))
    run_dir = Path(pipeline.get("artifact_root", "artifacts/runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_training_device(device, allow_cpu=allow_cpu)
    trial_specs = expand_grid_search(config_for_search)

    resolved_config_path = _write_yaml(run_dir / "resolved_config.yml", config_for_search)
    environment_path = _write_json(run_dir / "environment.json", _environment_manifest())
    run_identity = {
        **_config_identity(config_for_search, resolved_config_path),
        "source_config_path": config_path,
        "run_id": run_id,
        "run_dir": run_dir,
        "raw_data": _raw_data_manifest(config_for_search),
        "git": _git_manifest(),
        "dependencies": _dependency_manifest(),
        "device": _device_manifest(resolved_device),
        "trial_count": len(trial_specs),
        "environment_path": environment_path,
    }
    _write_json(run_dir / "run_identity.json", run_identity)
    stage0_dataset = build_multi_horizon_lighter_dataset(config_for_search, smoke=smoke)
    stage0 = _write_stage0(stage0_dataset, run_dir)
    benchmark_summary = None
    if (config_for_search.get("benchmark", {}) or {}).get("enabled", True):
        from src.experiments.forecast_benchmark import run_forecast_benchmark_from_dataset

        benchmark_summary = run_forecast_benchmark_from_dataset(
            config_for_search,
            stage0_dataset,
            run_dir / "forecast_benchmark",
            smoke=smoke,
            device=resolved_device,
            config_path=config_path,
            run_id=run_id,
        )

    trial_manifests = [_run_trial_safe(spec, run_dir, smoke=smoke, device=resolved_device) for spec in trial_specs]
    completed_trials = [trial for trial in trial_manifests if trial.get("status") == "completed"]
    if not completed_trials:
        trial_accounting = _trial_accounting(trial_manifests, {"min_validation_trades": int((config_for_search.get("pipeline", {}) or {}).get("min_validation_trades", 1) or 0)}, config_for_search)
        _write_json(run_dir / "trial_accounting.json", trial_accounting)
        raise RuntimeError("All v2 trial specs failed; see trial manifests for failure reasons")
    best, best_trading, selection = select_trials_with_trade_floor(completed_trials, config_for_search)
    best_path = run_dir / "best_trial_manifest.json"
    _write_json(best_path, best)
    best_trading_path = None
    if best_trading is not None:
        best_trading_path = run_dir / "best_trade_qualified_trial_manifest.json"
        _write_json(best_trading_path, best_trading)

    trial_accounting = _trial_accounting(trial_manifests, selection, config_for_search)
    trial_accounting_path = _write_json(run_dir / "trial_accounting.json", trial_accounting)

    for trial in trial_manifests:
        _log_meta_label_trial_trackio(
            config_for_search,
            run_id=run_id,
            trial=trial,
            selection_status=_selection_status_for_trial(trial, best, best_trading, selection),
            raw_best=best,
            best_trading=best_trading,
            selection=selection,
            smoke=smoke,
        )

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "run_dir": run_dir,
        "config_path": config_path,
        "smoke": smoke,
        "device": str(resolved_device),
        "run_identity": run_identity,
        "raw_data": run_identity["raw_data"],
        "trial_count": len(trial_specs),
        "stage0": stage0,
        "benchmark": benchmark_summary,
        "trials": trial_manifests,
        "best_trial": best,
        "best_trial_manifest_path": best_path,
        "best_trading_trial": best_trading,
        "best_trading_trial_manifest_path": best_trading_path,
        "selection": selection,
        "trial_accounting": trial_accounting,
        "trial_accounting_path": trial_accounting_path,
        "status": "success",
        "report_json_path": report_dir / "report.json",
        "report_markdown_path": report_dir / "report.md",
    }
    _write_json(report["report_json_path"], report)
    Path(report["report_markdown_path"]).write_text(_report_markdown(report), encoding="utf-8")
    _write_json(run_dir / "manifest.json", report)
    _log_meta_label_summary_trackio(config_for_search, run_id=run_id, report=report, smoke=smoke)
    return _json_ready(report)
