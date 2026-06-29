"""MVP hierarchical meta-labeling research pipeline.

This module implements the v2 config path used by ``configs/config.yml``. It is
kept separate from ``staged_trial.py`` so legacy staged GLFT experiments remain
runnable while the active config moves to multi-horizon forecasting, true
meta-labels, and directional alpha backtesting.
"""

from __future__ import annotations

import json
import math
import platform
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
from src.training.devices import resolve_training_device
from src.training.trainer import compute_metrics


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


def build_multi_horizon_lighter_dataset(config: Mapping[str, Any], *, smoke: bool = False) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    granularity = data_cfg.get("granularity", data_cfg.get("lighter", {}).get("resolution", DEFAULT_GRANULARITY))
    training_cfg = config.get("training", {})
    sequence_length = int(training_cfg.get("sequence_length", bars_for_duration(granularity, hours=24)))
    horizons = target_horizons(config)
    max_horizon = max(bars for _, bars in horizons)
    label_cfg = config.get("labels", {}).get("meta_triple_barrier", {})
    volatility_window = int(label_cfg.get("volatility_window", bars_for_duration(granularity, hours=24)))

    preprocessor = DataPreprocessor(data_dir=data_cfg.get("dir", "data"), granularities=[granularity])
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

    scaled_features = _normalise_features(features_df)
    X: List[torch.Tensor] = []
    y_ret: List[List[float]] = []
    y_dir: List[List[float]] = []
    sample_rows: List[Dict[str, Any]] = []
    last_start = len(features_df) - sequence_length - max_horizon
    for start in range(last_start):
        label_idx = start + sequence_length
        X.append(scaled_features[start:label_idx])
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
            }
        )

    path_columns = [col for col in ["open", "high", "low", "close", "volume"] if col in features_df.columns]
    price_path = features_df[path_columns].copy()
    price_path["timestamp"] = features_df.index
    price_path["realized_vol"] = realized_vol.to_numpy(dtype=float)
    if "vol_regime" in features_df.columns:
        price_path["vol_regime"] = features_df["vol_regime"].to_numpy(dtype=float)
    price_path = price_path.reset_index(drop=True)

    sample_weights = torch.full((len(X),), 1.0 / max(len(X), 1), dtype=torch.float32)
    return {
        "target": "multi_horizon",
        "X": torch.stack(X),
        "y_ret": torch.tensor(y_ret, dtype=torch.float32),
        "y_dir": torch.tensor(y_dir, dtype=torch.float32),
        "sample_weights": sample_weights,
        "samples": pd.DataFrame(sample_rows),
        "price_path": price_path,
        "close": pd.Series([row["close"] for row in sample_rows], index=pd.Index([row["timestamp"] for row in sample_rows], name="timestamp")),
        "input_size": int(scaled_features.shape[1]),
        "granularity": granularity,
        "sequence_length": sequence_length,
        "horizon_names": [name for name, _ in horizons],
        "horizon_bars": {name: bars for name, bars in horizons},
        "max_horizon_bars": max_horizon,
        "feature_columns": preprocessor.get_feature_cols(),
        "feature_window_bars": preprocessor.feature_window_bars(granularity),
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
    return model, {"training_losses": losses, "final_loss": losses[-1] if losses else 0.0}


def predict_base_model(model: MultiHorizonLSTM, dataset: Mapping[str, Any], indices: np.ndarray, device: torch.device, batch_size: int = 512) -> pd.DataFrame:
    X = dataset["X"][indices]
    pred_returns: List[torch.Tensor] = []
    pred_probs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size].to(device)
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


def add_meta_labels(candidates: pd.DataFrame, dataset: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    return meta_triple_barrier_labels(
        candidates,
        dataset["price_path"],
        config.get("labels", {}).get("meta_triple_barrier", {}),
        config.get("costs", {}),
        granularity=str(dataset["granularity"]),
    )


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
    if len(np.unique(y)) < 2:
        return ConstantMetaModel(float(np.mean(y)), feature_columns)
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
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=int(training.get("batch_size", 32)),
        shuffle=True,
    )
    for _ in range(int(training.get("epochs", 1))):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    return TorchMetaModel(model, mean, std, feature_columns, device)


def add_meta_probabilities(candidates: pd.DataFrame, meta_model: Any) -> pd.DataFrame:
    out = candidates.copy()
    out["meta_prob"] = meta_model.predict_proba(out)
    out.loc[~out["is_candidate"], "meta_prob"] = 0.0
    return out


def run_alpha_backtest(candidates: pd.DataFrame, config: Mapping[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    alpha = config.get("alpha_backtest", {}) or {}
    initial_capital = float(alpha.get("initial_capital", 100_000.0))
    meta_threshold = float(alpha.get("meta_threshold", 0.5))
    edge_threshold = float(alpha.get("edge_threshold_bps", 0.0))
    horizon_choice = str(alpha.get("horizon", "best"))
    notional = float(alpha.get("position_notional", initial_capital))
    max_notional = float(alpha.get("max_position_notional", notional))
    notional = min(notional, max_notional) if max_notional > 0 else notional

    eligible = candidates[
        (candidates["is_candidate"])
        & (candidates["meta_prob"] >= meta_threshold)
        & (candidates["expected_edge_bps"] >= edge_threshold)
    ].copy()
    if horizon_choice != "best":
        eligible = eligible[eligible["horizon"] == horizon_choice].copy()
    if not eligible.empty:
        eligible["score"] = eligible["meta_prob"] * eligible["expected_edge_bps"]
        eligible = eligible.sort_values(["timestamp", "score"], ascending=[True, False])
        eligible = eligible.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp")

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
        metrics = {
            "coverage": 0.0,
            "candidate_coverage": candidate_count / max(len(candidates), 1),
            "trades": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "fees": 0.0,
            "turnover": 0.0,
            "exposure": 0.0,
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
        }
        return metrics, pd.DataFrame(columns=["timestamp", "horizon", "side", "notional", "net_pnl"])

    trades_df["cumulative_net_pnl"] = trades_df["net_pnl"].cumsum()
    equity = initial_capital + trades_df["cumulative_net_pnl"]
    drawdown = equity - equity.cummax()
    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]
    returns = trades_df["net_return"].to_numpy(dtype=float)
    metrics = {
        "coverage": float(len(trades_df) / total_rows),
        "candidate_coverage": float(candidate_count / max(len(candidates), 1)),
        "trades": float(len(trades_df)),
        "gross_pnl": float(trades_df["gross_pnl"].sum()),
        "net_pnl": float(trades_df["net_pnl"].sum()),
        "fees": float(trades_df["fees"].sum()),
        "turnover": float(trades_df["notional"].sum()),
        "exposure": float(trades_df["notional"].sum() / (initial_capital * total_rows)),
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
    }
    return metrics, trades_df


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
    _write_json(trial_dir / "split_manifest.json", _split_manifest(dataset, splits))

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
    meta_model = fit_meta_labeler(oof_candidates, config, dataset["horizon_names"], device)
    oof_candidates = add_meta_probabilities(oof_candidates, meta_model)
    test_candidates = add_meta_probabilities(test_candidates, meta_model)

    validation_metrics, validation_trades = run_alpha_backtest(oof_candidates, config)
    test_metrics, test_trades = run_alpha_backtest(test_candidates, config)

    model_path = trial_dir / "base_multi_horizon_lstm.pt"
    torch.save(final_model.state_dict(), model_path)
    paths = {
        "oof_predictions": trial_dir / "predictions_oof.parquet",
        "test_predictions": trial_dir / "predictions_test.parquet",
        "oof_candidates": trial_dir / "meta_candidates_oof.parquet",
        "test_candidates": trial_dir / "meta_candidates_test.parquet",
        "validation_trades": trial_dir / "alpha_trades_validation.parquet",
        "test_trades": trial_dir / "alpha_trades_test.parquet",
    }
    oof_predictions.to_parquet(paths["oof_predictions"], index=False)
    test_predictions.to_parquet(paths["test_predictions"], index=False)
    oof_candidates.to_parquet(paths["oof_candidates"], index=False)
    test_candidates.to_parquet(paths["test_candidates"], index=False)
    validation_trades.to_parquet(paths["validation_trades"], index=False)
    test_trades.to_parquet(paths["test_trades"], index=False)

    manifest = {
        "stage": "mvp_trial",
        "trial_index": int(spec["trial_index"]),
        "trial_id": trial_id,
        "overrides": spec.get("overrides", {}),
        "config": config,
        "dataset": {
            "samples": int(dataset["X"].shape[0]),
            "input_size": int(dataset["input_size"]),
            "sequence_length": int(dataset["sequence_length"]),
            "horizons": dataset["horizon_bars"],
        },
        "folds": fold_manifests,
        "final_training_history": final_history,
        "meta_labeler": meta_model.manifest(),
        "model_path": model_path,
        "artifact_paths": paths,
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "manifest_path": trial_dir / "manifest.json",
    }
    _write_json(trial_dir / "manifest.json", manifest)
    return manifest


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


def _report_markdown(report: Mapping[str, Any]) -> str:
    best = report["best_trial"]
    lines = [
        f"# Meta-Labeling MVP Report: {report['run_id']}",
        "",
        f"- Status: `{report['status']}`",
        f"- Trials: `{len(report['trials'])}`",
        f"- Best trial: `{best['trial_id']}`",
        f"- Validation net PnL: `{best['metrics']['validation']['net_pnl']}`",
        f"- Test net PnL: `{best['metrics']['test']['net_pnl']}`",
        f"- Test trades: `{best['metrics']['test']['trades']}`",
        f"- Test coverage: `{best['metrics']['test']['coverage']}`",
        "",
        "## Artifacts",
        "",
        f"- Report JSON: `{report['report_json_path']}`",
        f"- Best trial manifest: `{best['manifest_path']}`",
    ]
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

    pipeline = config_for_search.get("pipeline", {}) or {}
    run_id = _run_id(str(pipeline.get("run_name", "meta_label_mvp")))
    run_dir = Path(pipeline.get("artifact_root", "artifacts/runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_training_device(device, allow_cpu=allow_cpu)

    _write_yaml(run_dir / "resolved_config.yml", config_for_search)
    _write_json(run_dir / "environment.json", _environment_manifest())
    stage0_dataset = build_multi_horizon_lighter_dataset(config_for_search, smoke=smoke)
    stage0 = _write_stage0(stage0_dataset, run_dir)

    trial_specs = expand_grid_search(config_for_search)
    trial_manifests = [_run_one_trial(spec, run_dir, smoke=smoke, device=resolved_device) for spec in trial_specs]
    ranked = _rank_trials(trial_manifests, config_for_search)
    best = dict(ranked[0])
    best_path = run_dir / "best_trial_manifest.json"
    _write_json(best_path, best)

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "run_dir": run_dir,
        "config_path": config_path,
        "smoke": smoke,
        "device": str(resolved_device),
        "stage0": stage0,
        "trials": trial_manifests,
        "best_trial": best,
        "best_trial_manifest_path": best_path,
        "status": "success",
        "report_json_path": report_dir / "report.json",
        "report_markdown_path": report_dir / "report.md",
    }
    _write_json(report["report_json_path"], report)
    Path(report["report_markdown_path"]).write_text(_report_markdown(report), encoding="utf-8")
    _write_json(run_dir / "manifest.json", report)
    return _json_ready(report)
