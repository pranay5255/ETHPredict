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
import time
import warnings
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
from src.config.search import build_search_plan
from src.evaluation.selection import ALIASES, validation_selection_scores, comparison_budget
from src.simulator.costs import dataset_cost_model, configured_notional
from src.data.features_all import DEFAULT_GRANULARITY, DataPreprocessor, bars_for_duration
from src.data.side_data import DataCoverageError, source_paths, validate_split_coverage
from src.evaluation.backtest_stats import (
    ANNUALISATION_NOTE,
    attach_cross_trial_statistics,
    classify_evidence_grade,
    period_pnl_by_timestamp,
    period_pnl_series,
    summarize_single_path,
)
from src.evaluation.forecast_diagnostics import calibration_summary, forecast_diagnostics, map_forecast
from src.features.labeling import meta_triple_barrier_labels
from src.features.sample_weights import (
    average_uniqueness,
    build_label_spans,
    sample_weight_config,
    sample_weight_diagnostics,
    sample_weights_and_diagnostics,
)
from src.validation.splits import exact_validation_splits, purge_training_spans
from src.training.meta_models import MetaModelBundle, SklearnMetaModel, fit_calibrator
from src.training.discipline import EarlyStopper, SeedEnsemble, TabularBase, bootstrap_rows, training_holdout
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
    def __init__(self, probability: float, feature_columns: Sequence[str], reason: str = "configured_constant"):
        self.probability = float(np.clip(probability, 0.0, 1.0))
        self.feature_columns = list(feature_columns)
        self.kind = "constant"
        self.reason = reason

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.probability, dtype=float)

    def manifest(self) -> Dict[str, Any]:
        return {"kind": self.kind, "probability": self.probability, "feature_columns": self.feature_columns, "fallback_reason": self.reason}


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
        return {"kind": self.kind, "feature_columns": self.feature_columns,
                "preprocessing": {"fit_scope": "training_rows_only", "mean": self.mean.tolist(), "std": self.std.tolist()},
                "training": getattr(self, "training_history", {})}


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


REPO_ROOT = Path(__file__).resolve().parents[2]
CRITICAL_PATH_PREFIXES = (
    "configs/",
    "src/",
    "tests/",
    "context/",
    "scripts/",
    "TASKS.md",
    "pyproject.toml",
    "uv.lock",
)
FEATURE_CODE_FILES = (
    "src/data/features_all.py",
    "src/data/side_data.py",
    "src/features/labeling.py",
    "src/features/sample_weights.py",
    "src/experiments/meta_labeling_mvp.py",
    "src/simulator/costs.py",
)
# Config sections that define a research specification. Naming and storage keys
# (pipeline.run_name, pipeline.artifact_root, experiment.id/name, tracking) are
# deliberately excluded so renaming a run cannot bypass the final-test guard.
RESEARCH_SPEC_SECTIONS = (
    "version",
    "data",
    "bars",
    "features",
    "sampling",
    "sample_weights",
    "targets",
    "labels",
    "model",
    "training",
    "validation",
    "costs",
    "alpha_backtest",
    "candidate_mapping",
    "feature_importance",
)
FINAL_TEST_SELECTION_ROLES = ("raw_best", "trade_qualified_best", "classification_best", "calibration_best")
DEFAULT_FINAL_TEST_ROLES = ("raw_best", "trade_qualified_best")
RAW_DATA_CHANGE_POLICIES = ("warn", "fail", "ignore")


def _git_status_entry(line: str) -> Tuple[str, str]:
    """Parse one ``git status --porcelain`` line into ``(status, path)``.

    Leading whitespace is significant in porcelain output (`` M path``), so the
    status column is split on the first run of whitespace instead of by offset.
    """

    stripped = line.strip()
    if not stripped:
        return "", ""
    parts = stripped.split(None, 1)
    status, path = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return status, path.strip().strip('"')


def _git_manifest() -> Dict[str, Any]:
    status = _command_output(["git", "status", "--porcelain"])
    critical = []
    for line in (status or "").splitlines():
        entry_status, path = _git_status_entry(line)
        if path.startswith(CRITICAL_PATH_PREFIXES):
            critical.append({"status": entry_status, "path": path})
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
    files += [path for _, path in source_paths(data_cfg.get("dir", "data"), data_cfg.get("side_data", {}))]
    funding_source = config.get("costs", {}).get("dynamic_funding", {})
    if config.get("costs", {}).get("mode") == "dynamic" and funding_source.get("enabled"):
        for value in funding_source.get("paths", []):
            path = Path(value)
            files.append(path if path.is_absolute() else Path(data_cfg.get("dir", "data")) / path)
    files = list(dict.fromkeys(files))
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


def research_spec(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the research-relevant part of a config.

    Only data, features, labels, model, split, costs, policy, and seed are kept.
    Run names, artifact roots, experiment labels, tracking, and search metadata
    are excluded so they cannot change the identity of a research specification.
    """

    spec = {key: deepcopy(config[key]) for key in RESEARCH_SPEC_SECTIONS if key in config}
    spec["seed"] = (config.get("experiment", {}) or {}).get("seed")
    spec["selection_policy"] = _selection_policy(config)
    return spec


def _selection_policy(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Trading-selection policy, without run names or artifact locations."""

    pipeline = config.get("pipeline") or {}
    min_trades = pipeline.get("min_validation_trades", pipeline.get("selection_min_trades", 1))
    return {
        "selection_metric": pipeline.get("selection_metric", "metrics.validation.net_pnl"),
        "selection_mode": pipeline.get("selection_mode", "max"),
        "min_validation_trades": min_trades,
    }


def research_spec_hash(config: Mapping[str, Any], **extra: Any) -> str:
    return _stable_hash({"research_spec": research_spec(config), **extra})


def _research_cfg(config: Mapping[str, Any], key: str) -> Dict[str, Any]:
    return dict(((config.get("research", {}) or {}).get(key, {}) or {}))


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _final_test_ledger_path(config: Mapping[str, Any], run_dir: Path) -> Path:
    configured = _research_cfg(config, "final_test_guard").get("ledger_path")
    return Path(configured) if configured else run_dir.parent / "_final_test_reuse_ledger.json"


def final_test_use_count(trials: Sequence[Mapping[str, Any]]) -> int:
    """Highest ledger count of the specs evaluated in this run, read after scoring.

    The count stored on the first selected trial is the value at that trial's
    evaluation. A later selected trial of the same spec increments the ledger, so
    the grade must read the ledger again.
    """

    seen = set()
    counts: List[int] = []
    for trial in trials:
        evaluation = trial.get("final_test_evaluation") or {}
        if not isinstance(evaluation, Mapping) or not evaluation.get("spec_hash"):
            continue
        spec_hash = str(evaluation["spec_hash"])
        ledger_path = evaluation.get("ledger_path")
        key = (spec_hash, str(ledger_path))
        if key in seen:
            continue
        seen.add(key)
        if ledger_path:
            record = _read_json_file(Path(ledger_path)).get(spec_hash, {})
            if isinstance(record, Mapping) and record.get("count") is not None:
                counts.append(int(record["count"]))
                continue
        if evaluation.get("count") is not None:
            counts.append(int(evaluation["count"]))
    return max(counts) if counts else 0


def final_test_selection_roles(config: Mapping[str, Any]) -> List[str]:
    """Selection roles whose trials may be evaluated on the final test split."""

    if _research_cfg(config, "final_test_guard").get("evaluate", True) is False:
        return []
    roles = _research_cfg(config, "final_test_guard").get("evaluate_selection_roles", list(DEFAULT_FINAL_TEST_ROLES))
    roles = [str(role) for role in (roles or [])]
    unknown = sorted(set(roles) - set(FINAL_TEST_SELECTION_ROLES))
    if unknown:
        raise ValueError(f"Unknown research.final_test_guard.evaluate_selection_roles {unknown}; allowed={list(FINAL_TEST_SELECTION_ROLES)}")
    if not roles:
        raise ValueError("research.final_test_guard.evaluate_selection_roles must name at least one selection role")
    return roles


def _record_final_test_evaluation(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    trial_id: str,
    split_hash: str,
    smoke: bool,
    scope: str = "meta_label_trial",
) -> Dict[str, Any]:
    """Count one final-test query for a research spec and enforce the reuse guard.

    Call this before test data is scored. The spec hash covers only the research
    specification (see ``research_spec``), the split identity, and the evaluation
    scope, so renaming a run or moving its artifact directory is still reuse.
    """

    guard_cfg = _research_cfg(config, "final_test_guard")
    spec_hash = research_spec_hash(config, split_hash=split_hash, scope=scope)
    ledger_path = _final_test_ledger_path(config, run_dir)
    ledger = _read_json_file(ledger_path)
    record = dict(ledger.get(spec_hash, {}))
    count = int(record.get("count", 0)) + 1
    do_not_reuse = bool(guard_cfg.get("do_not_reuse_test", config.get("do_not_reuse_test", False)))
    blocked = do_not_reuse and count > 1 and not smoke
    evaluated_at = _utc_now().isoformat()
    history = list(record.get("history", []) or [])
    history.append({"trial_id": trial_id, "run_dir": str(run_dir), "evaluated_at_utc": evaluated_at, "smoke": bool(smoke), "blocked": blocked})
    record.update(
        {
            "count": count,
            "scope": scope,
            "last_trial_id": trial_id,
            "last_run_dir": str(run_dir),
            "split_hash": split_hash,
            "history": history,
        }
    )
    ledger[spec_hash] = record
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    payload = {
        "spec_hash": spec_hash,
        "scope": scope,
        "count": count,
        "ledger_path": ledger_path,
        "blocked": blocked,
        "do_not_reuse_test": do_not_reuse,
        "evaluated_at_utc": evaluated_at,
    }
    if blocked:
        raise RuntimeError(
            f"Final test set reuse blocked for research spec {spec_hash} ({scope}); ledger count={count}. "
            "Declare a new research specification before evaluating the final test split again."
        )
    return payload


def raw_data_fingerprint(raw_manifest: Mapping[str, Any]) -> str:
    files = raw_manifest.get("files", []) or []
    return _stable_hash(sorted((str(item.get("path")), str(item.get("sha256"))) for item in files))


def check_raw_data_registry(
    config: Mapping[str, Any],
    artifact_root: Path,
    *,
    run_id: str,
    raw_manifest: Mapping[str, Any],
    smoke: bool,
) -> Dict[str, Any]:
    """Detect raw-data changes under an unchanged research spec at run start.

    ``research.raw_data_guard.on_change`` selects ``warn`` (default), ``fail``, or
    ``ignore``. Smoke runs never fail; they record the change as a warning.
    """

    guard_cfg = _research_cfg(config, "raw_data_guard")
    policy = str(guard_cfg.get("on_change", "warn"))
    if policy not in RAW_DATA_CHANGE_POLICIES:
        raise ValueError(f"research.raw_data_guard.on_change must be one of {list(RAW_DATA_CHANGE_POLICIES)}; got {policy!r}")
    registry_path = Path(guard_cfg["registry_path"]) if guard_cfg.get("registry_path") else Path(artifact_root) / "_raw_data_registry.json"
    spec_hash = research_spec_hash(config, scope="raw_data")
    fingerprint = raw_data_fingerprint(raw_manifest)
    registry = _read_json_file(registry_path)
    record = dict(registry.get(spec_hash, {}))
    previous = record.get("fingerprint")
    if previous is None:
        status = "first_seen"
    elif previous == fingerprint:
        status = "unchanged"
    else:
        status = "changed"
    effective_policy = "warn" if smoke and policy == "fail" else policy
    result = {
        "status": status,
        "policy": policy,
        "effective_policy": effective_policy,
        "spec_hash": spec_hash,
        "fingerprint": fingerprint,
        "previous_fingerprint": previous,
        "previous_run_id": record.get("last_run_id"),
        "registry_path": registry_path,
    }
    if status == "changed" and effective_policy == "fail":
        raise RuntimeError(
            f"Raw data changed under research spec {spec_hash}: fingerprint {previous} -> {fingerprint} "
            f"(previous run {record.get('last_run_id')}). Declare a new research spec or set "
            "research.raw_data_guard.on_change to warn."
        )
    if status == "changed" and effective_policy == "warn":
        warnings.warn(
            f"Raw data changed under research spec {spec_hash} since run {record.get('last_run_id')}; "
            "results are not comparable with earlier runs of this spec.",
            RuntimeWarning,
            stacklevel=2,
        )
    history = list(record.get("history", []) or [])
    history.append({"run_id": run_id, "fingerprint": fingerprint, "status": status, "checked_at_utc": _utc_now().isoformat()})
    record.update({"fingerprint": fingerprint, "last_run_id": run_id, "history": history, "files": raw_manifest.get("files", [])})
    registry[spec_hash] = record
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(_json_ready(registry), indent=2, sort_keys=True), encoding="utf-8")
    return result


def feature_code_identity(root: Optional[Path] = None) -> Dict[str, Any]:
    """Hash the source files that define v2 features, labels, and sample weights."""

    base = Path(root) if root is not None else REPO_ROOT
    files: Dict[str, Optional[str]] = {}
    for relative in FEATURE_CODE_FILES:
        path = base / relative
        files[relative] = _sha256_path(path) if path.exists() else None
    return {"files": files, "hash": _stable_hash(files)}


def _feature_manifest_with_identity(feature_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = dict(feature_manifest)
    fracdiff = manifest.get("fracdiff", {}) or {}
    # Only configuration-level fracdiff fields; data-dependent diagnostics
    # (ADF p-value, row counts) must not change the family identity.
    manifest["family_identity_hash"] = _stable_hash(
        {
            "families": manifest.get("families", []),
            "columns": manifest.get("columns", []),
            "fracdiff": {key: fracdiff.get(key) for key in ("mode", "order", "threshold")},
        }
    )
    manifest["code_identity"] = feature_code_identity()
    return manifest


def validate_selection_metric(config: Mapping[str, Any]) -> str:
    """Reject selection metrics that read final-test results."""

    metric = str((config.get("pipeline", {}) or {}).get("selection_metric", "metrics.validation.net_pnl"))
    metric = ALIASES.get(metric, metric)
    if not metric.startswith("metrics.validation."):
        raise ValueError("Trial selection requires validation evidence; final-test and non-validation metrics are prohibited")
    parts = metric.split(".")
    if "test" in parts:
        raise ValueError(
            f"pipeline.selection_metric={metric!r} reads final-test metrics; trial selection must use validation evidence only"
        )
    return metric


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
    max_horizon = max([max_horizon, *map(int, (label_cfg.get("vertical_barriers", {}) or {}).values())])
    max_horizon += int(config.get("costs", {}).get("execution", {}).get("entry_delay_bars", 0))
    volatility_window = int(label_cfg.get("volatility_window", bars_for_duration(granularity, hours=24)))

    bar_cfg = dict(config.get("bars", {}) or data_cfg.get("bars", {}) or {})
    if data_cfg.get("bar_type") is not None:
        bar_cfg["type"] = data_cfg.get("bar_type")
    preprocessor = DataPreprocessor(
        data_dir=data_cfg.get("dir", "data"),
        granularities=[granularity],
        feature_config=config.get("features", {}) or {},
        bar_config=bar_cfg,
        side_data_config=data_cfg.get("side_data", {}),
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

    price_frame = preprocessor.price_frame(granularity).set_index("timestamp").reindex(features_df.index)
    prices = targets_df["close"].astype(float)
    log_returns = np.log(prices.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    realized_vol = log_returns.rolling(volatility_window, min_periods=2).std()
    # Warmup must not depend on volatility observed later in the dataset.
    realized_vol = realized_vol.fillna(0.001).clip(lower=1e-8)

    importance_cfg = config.get("feature_importance", {}) or {}
    noise_count = int(importance_cfg.get("noise_features", 0)) if importance_cfg.get("enabled", False) else 0
    if noise_count < 0:
        raise ValueError("noise_features cannot be negative")
    if noise_count:
        features_df = features_df.copy()
        noise = np.random.default_rng(int(importance_cfg.get("seed", 42))).normal(size=(len(features_df), noise_count))
        for i in range(noise_count):
            features_df[f"diagnostic_noise_{i}"] = noise[:, i]
    feature_manifest = preprocessor.feature_manifest()
    feature_manifest["columns"] = list(features_df.columns)
    if noise_count:
        feature_manifest["families"] = feature_manifest.get("families", []) + ["diagnostic_noise"]
        feature_manifest["noise_controls"] = {"count": noise_count, "seed": int(importance_cfg.get("seed", 42))}
    raw_features = torch.tensor(features_df.to_numpy(dtype=float), dtype=torch.float32)
    X: List[torch.Tensor] = []
    y_ret: List[List[float]] = []
    y_dir: List[List[float]] = []
    sample_rows: List[Dict[str, Any]] = []
    dense_label_indices = list(range(sequence_length, len(features_df) - max_horizon))
    if data_cfg.get("sample_start"):
        sample_start = pd.Timestamp(data_cfg["sample_start"])
        if sample_start.tz is not None:
            sample_start = sample_start.tz_convert("UTC").tz_localize(None)
        dense_label_indices = [index for index in dense_label_indices if features_df.index[index] >= sample_start]
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
        price_row = price_frame.iloc[label_idx]
        horizon_span_meta: Dict[str, Any] = {}
        for horizon_name, horizon_bars in horizons:
            horizon_end_idx = int(label_idx + horizon_bars)
            horizon_span_meta[f"{horizon_name}_label_span_end_idx"] = horizon_end_idx
            horizon_span_meta[f"{horizon_name}_label_span_end_timestamp"] = prices.index[horizon_end_idx]
        sample_rows.append(
            {
                "timestamp": prices.index[label_idx],
                "bar_close_timestamp": pd.to_datetime(price_row["close_time"], unit="ms"),
                "sample_index": label_idx,
                "close": entry,
                "open": float(price_row["open"]),
                "high": float(price_row["high"]),
                "low": float(price_row["low"]),
                "volume": float(price_row["volume"]),
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

    price_path = price_frame[["open_time", "open", "high", "low", "close", "volume", "close_time"]].copy()
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
        "feature_columns": list(features_df.columns),
        "feature_window_bars": preprocessor.feature_window_bars(granularity),
        "feature_manifest": _feature_manifest_with_identity(feature_manifest),
        "bar_manifest": preprocessor.bar_manifest(),
        "event_diagnostics": event_diagnostics,
        "sample_weight_diagnostics": sample_weight_diag,
        "bar_clock_diagnostics": bar_clock_diagnostics,
        "side_data_manifest": preprocessor.side_data_manifest(),
        "feature_frame": features_df,
        "source_timestamps": preprocessor.source_timestamps(granularity),
        "coverage_config": {"bars": bar_cfg, "side_data": data_cfg.get("side_data", {}),
                            "source_interval_ms": int(pd.Timedelta(granularity).total_seconds() * 1000)},
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


def build_validation_splits(dataset: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    """One split contract for v2 training and every forecast benchmark.

    Legacy fixed gaps remain available, with an explicit final safety guard
    when configured gaps are shorter than observed inclusive label spans.
    """
    cfg = config.get("validation", {})
    if cfg.get("method", "purged_walk_forward") != "purged_walk_forward":
        splits = exact_validation_splits(dataset, cfg)
        splits["data_coverage"] = validate_split_coverage(dataset, splits)
        return splits
    splits = purged_walk_forward_splits(len(dataset["X"]), cfg)
    clock = dataset["price_path"]["timestamp"]
    guard = {"embargo_bars": 0}
    development, audit = purge_training_spans(dataset["samples"], splits["development"], splits["test"], guard, clock=clock)
    if len(development) < 2:
        raise ValueError("Insufficient development coverage after label-span safety guard")
    splits["development"] = development
    splits["gap"] = np.setdiff1d(np.arange(splits["test_start"]), development)
    splits["final_boundary_audit"] = audit
    for fold in splits["folds"]:
        fold["validation"] = np.intersect1d(fold["validation"], development)
        if not len(fold["validation"]):
            raise ValueError("Insufficient validation coverage after final-test span protection")
        fold["train"], fold["audit"] = purge_training_spans(dataset["samples"], fold["train"], fold["validation"], guard, clock=clock)
        if not len(fold["train"]):
            raise ValueError("Insufficient training coverage after fixed-gap label-span safety guard")
    splits.update(method="purged_walk_forward", path_type="walk_forward", span_guard=True)
    splits["data_coverage"] = validate_split_coverage(dataset, splits)
    return splits


def fit_outer_meta_model(dataset, training_indices, config, device, prediction_provider=None, evaluation_spans=None):
    """Nested base OOF predictions from the outer training population only.

    In purged k-fold/grouped scenarios even another outer fold's base model
    may have seen the current evaluation labels. Inner base fits prevent that
    indirect route into meta training.
    """
    count = int(config.get("validation", {}).get("meta_inner_folds", 2))
    if count < 1:
        raise ValueError("meta_inner_folds must be positive")
    groups = np.array_split(np.asarray(training_indices, dtype=int), count + 1)
    frames, audits = [], []
    clock = dataset["price_path"]["timestamp"]
    for index in range(1, len(groups)):
        validation = groups[index]
        proposed = np.concatenate(groups[:index])
        if not len(validation) or not len(proposed):
            continue
        train, audit = purge_training_spans(dataset["samples"], proposed, validation,
                                             config.get("validation", {}), clock=clock,
                                             max_horizon=dataset["max_horizon_bars"], sequence_length=dataset["sequence_length"])
        if not len(train):
            audits.append({**audit, "status": "unavailable", "reason": "no inner training rows after purging"})
            continue
        audit["data_coverage"] = validate_split_coverage(dataset, {"folds": [
            {"fold": index, "train": train, "validation": validation}
        ]}, include_final=False)
        if prediction_provider is None:
            model, _ = train_base_model(dataset, train, config, device)
            predictions = predict_base_model(model, dataset, validation, device)
            predictions["prediction_role"] = "base_oof"
        else:
            predictions = prediction_provider(validation)
            if predictions.empty:
                continue
            predictions["prediction_role"] = "untrained_baseline"
        frames.append(add_meta_labels(candidate_signals(predictions, dataset, config), dataset, config))
        audits.append({**audit, "status": "completed"})
    if not frames:
        return ConstantMetaModel(0.0, [], "no_nested_base_oof_training_candidates"), {"status": "unavailable", "reason": "no nested base OOF training candidates", "inner_folds": audits}
    candidates = pd.concat(frames, ignore_index=True)
    realized_audit = None
    if evaluation_spans is not None and len(evaluation_spans):
        combined = pd.concat([candidates, evaluation_spans], ignore_index=True)
        retained, realized_audit = purge_training_spans(
            combined, np.arange(len(candidates)), np.arange(len(candidates), len(combined)),
            config.get("validation", {}), clock=clock,
            max_horizon=dataset["max_horizon_bars"], sequence_length=dataset["sequence_length"],
        )
        realized_audit["span_kind"] = "meta_realized_t1"
        realized_audit["candidate_keys"] = combined[["sample_index", "horizon", "t1_idx"]].to_dict("records")
        candidates = candidates.iloc[retained].copy()
    model = fit_meta_labeler(candidates, config, dataset["horizon_names"], device, dataset=dataset)
    labeled_count = int((candidates["is_candidate"] & candidates["meta_label"].notna()).sum())
    return model, {"status": "completed" if labeled_count else "unavailable",
                   "reason": None if labeled_count else "no labeled nested training candidates",
                   "labeled_candidate_count": labeled_count, "model": model.manifest(), "outer_train_indices": list(map(int, training_indices)),
                   "candidate_count": len(candidates), "inner_folds": audits,
                   "realized_t1_purge": realized_audit}


def evaluate_meta_fold(predictions, dataset, fold, config, device, prediction_provider=None, *, return_model=False):
    candidates = add_meta_labels(candidate_signals(predictions, dataset, config), dataset, config)
    model, audit = fit_outer_meta_model(dataset, fold["train"], config, device, prediction_provider, candidates)
    candidates = add_meta_probabilities(candidates, model)
    candidates["meta_evaluation_status"] = "held_out" if audit["status"] == "completed" else "unavailable"
    candidates["meta_fold"] = int(fold["fold"])
    return (candidates, audit, model) if return_model else (candidates, audit)


def collapse_scenario_predictions(frame, *, candidates=False):
    """Use the first predeclared scenario per row; retain full scenario reports.

    Never average probabilities of proposals with different sides or eligibility.
    This deterministic summary is not a stitched CPCV path.
    """
    keys = ["sample_index"] + (["horizon"] if candidates else [])
    priority = "meta_fold" if candidates and "meta_fold" in frame else "fold"
    if priority in frame:
        frame = frame.sort_values(priority, kind="stable")
    return frame.drop_duplicates(keys).sort_values("timestamp").reset_index(drop=True)


def _subset(dataset: Mapping[str, Any], indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return dataset["X"][indices], dataset["y_ret"][indices], dataset["y_dir"][indices], dataset["sample_weights"][indices]


def _bootstrap_audit(mode: str, draws: np.ndarray) -> Dict[str, Any]:
    # Without resampling the draws are just the retained rows in order.
    audit = {"mode": mode, "draw_count": int(len(draws)), "unique_rows": int(len(np.unique(draws)))}
    if mode != "none":
        audit["draws"] = draws.tolist()
    return audit


def _train_base_member(dataset: Mapping[str, Any], indices: np.ndarray, config: Mapping[str, Any], device: torch.device) -> Tuple[MultiHorizonLSTM, Dict[str, Any]]:
    training = config.get("training", {})
    stop_cfg = training.get("early_stopping", {}) or {}
    stop_audit = {"status": "disabled"}
    fit_indices, validation = np.asarray(indices), np.array([], dtype=int)
    if stop_cfg.get("enabled", False) and config.get("model", {}).get("base", {}).get("type", "lstm") == "lstm":
        fit_indices, validation, stop_audit = training_holdout(
            dataset["samples"], indices, config, stop_cfg, dataset=dataset)
    indices = fit_indices
    X, y_ret, y_dir, weights = _subset(dataset, indices)
    X = X.to(device)
    feature_mean = X.mean(dim=(0, 1), keepdim=True)
    feature_std = X.std(dim=(0, 1), keepdim=True, unbiased=False) + 1e-8
    X = (X - feature_mean) / feature_std
    y_ret = y_ret.to(device)
    y_dir = y_dir.to(device)
    weight_cfg = sample_weight_config(config)
    train_weights, weight_diagnostics = sample_weights_and_diagnostics(
        dataset["samples"].iloc[indices], mode=weight_cfg["base_mode"], normalize=weight_cfg["normalize"],
        return_values=dataset["y_ret"][indices].cpu().numpy(),
    )
    weights = torch.tensor(train_weights.to_numpy(), dtype=torch.float32, device=device)
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
    # Persist training-only preprocessing with the checkpoint state.
    model.register_buffer("feature_mean", feature_mean.detach().clone())
    model.register_buffer("feature_std", feature_std.detach().clone())
    seed = int(torch.initial_seed())
    draws = bootstrap_rows(dataset["samples"].iloc[indices], str(training.get("bootstrap", "none")), seed)
    kind = str(model_cfg.get("type", "lstm"))
    if kind in {"linear", "tree"}:
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        matrix = TabularBase.summary(X.detach().cpu().numpy())[draws]
        targets, directions = y_ret.cpu().numpy()[draws], y_dir.cpu().numpy()[draws]
        regressors, classifiers = [], []
        for h in range(targets.shape[1]):
            reg = (Ridge(alpha=float(model_cfg.get("alpha", 1.))) if kind == "linear" else
                   RandomForestRegressor(n_estimators=int(model_cfg.get("n_estimators", 50)),
                                         max_depth=int(model_cfg.get("max_depth", 4)), random_state=seed))
            reg.fit(matrix, targets[:, h], sample_weight=weights.detach().cpu().numpy()[draws] / max(float(weights.mean().cpu()), 1e-12))
            regressors.append(reg)
            if len(np.unique(directions[:, h])) < 2:
                classifiers.append(float(directions[:, h].mean()))
            else:
                clf = (LogisticRegression(max_iter=1000, random_state=seed) if kind == "linear" else
                       RandomForestClassifier(n_estimators=int(model_cfg.get("n_estimators", 50)),
                                              max_depth=int(model_cfg.get("max_depth", 4)), random_state=seed))
                clf.fit(matrix, directions[:, h], sample_weight=weights.detach().cpu().numpy()[draws] / max(float(weights.mean().cpu()), 1e-12))
                classifiers.append(clf)
        model = TabularBase(regressors, classifiers).to(device)
        model.register_buffer("feature_mean", feature_mean.detach().clone())
        model.register_buffer("feature_std", feature_std.detach().clone())
        return model, {"model_type": kind, "feature_summary": "latest_mean_std", "seed": seed,
                       "early_stopping": {"status": "not_applicable", "reason": "noniterative tabular fit"},
                       "sample_weight_diagnostics": weight_diagnostics,
                       "bootstrap": _bootstrap_audit(training.get("bootstrap", "none"), draws),
                       "preprocessing": {"fit_scope": "train_indices_only"}}
    if kind != "lstm":
        raise ValueError(f"Unknown base model type: {kind}")
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    ret_loss = nn.MSELoss(reduction="none")
    dir_loss = nn.BCEWithLogitsLoss(reduction="none")
    loader = DataLoader(
        TensorDataset(X[draws], y_ret[draws], y_dir[draws], weights[draws]),
        batch_size=int(training.get("batch_size", 32)),
        shuffle=True,
    )
    losses: List[float] = []
    validation_losses = []
    stopper = EarlyStopper(stop_cfg.get("patience", 3), stop_cfg.get("min_delta", 0.))
    for epoch in range(int(training.get("epochs", 1))):
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
        if len(validation):
            model.eval()
            with torch.no_grad():
                vx = (dataset["X"][validation].to(device) - feature_mean) / feature_std
                vr, vd = model(vx)
                vl = (float(loss_weights.get("return", 1.)) * ret_loss(vr, dataset["y_ret"][validation].to(device)).mean()
                      + float(loss_weights.get("direction", .25)) * dir_loss(vd, dataset["y_dir"][validation].to(device)).mean())
            validation_losses.append(float(vl.cpu()))
            if stopper.update(validation_losses[-1], epoch + 1, model):
                break
    stopper.restore(model)
    stop_audit.update(selected_epoch=stopper.best_epoch if len(validation) else len(losses),
                      validation_losses=validation_losses, epochs_run=len(losses),
                      best_loss=stopper.best_loss if len(validation) else None)
    return model, {
        "model_type": kind, "seed": seed, "early_stopping": stop_audit,
        "bootstrap": _bootstrap_audit(training.get("bootstrap", "none"), draws),
        "sample_weight_fit_scope": "retained_training_rows_only",
        "sample_weight_diagnostics": weight_diagnostics,
        "training_losses": losses,
        "final_loss": losses[(stopper.best_epoch or len(losses)) - 1] if losses else 0.0,
        "preprocessing": {
            "scaler": "standard",
            "fit_scope": "train_indices_only",
            "mean_shape": list(feature_mean.shape),
            "std_shape": list(feature_std.shape),
        },
    }


def train_base_model(dataset, indices, config, device):
    seeds = config.get("training", {}).get("ensemble_seeds")
    if seeds is None:
        seeds = [int(torch.initial_seed()) % (2**32)]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("ensemble_seeds must be nonempty and unique")
    members, histories = [], []
    for seed in seeds:
        torch.manual_seed(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(int(seed))
        model, history = _train_base_member(dataset, indices, config, device)
        members.append(model)
        histories.append(history)
    if len(members) == 1:
        return members[0], histories[0]
    return SeedEnsemble(members, seeds), {"ensemble_seeds": list(seeds), "aggregation": "arithmetic_mean_return_and_probability",
                                         "members": histories}


def predict_base_model(model: MultiHorizonLSTM, dataset: Mapping[str, Any], indices: np.ndarray, device: torch.device, batch_size: int = 512) -> pd.DataFrame:
    if isinstance(model, SeedEnsemble):
        frames = [predict_base_model(member, dataset, indices, device, batch_size) for member in model.members]
        frame = frames[0].copy()
        for horizon in dataset["horizon_names"]:
            for suffix in ["pred_return", "direction_prob"]:
                key = f"{horizon}_{suffix}"
                values = np.stack([part[key].to_numpy() for part in frames])
                frame[key] = values.mean(axis=0)
                frame[f"{key}_seed_std"] = values.std(axis=0)
                for seed, row in zip(model.seeds, values):
                    frame[f"{key}_seed_{seed}"] = row
        return frame
    if not len(indices):
        frame = dataset["samples"].iloc[:0].copy()
        for horizon in dataset["horizon_names"]:
            for suffix in ["pred_return", "direction_prob", "true_return", "true_direction"]:
                frame[f"{horizon}_{suffix}"] = np.array([], dtype=float)
        return frame
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
            **forecast_diagnostics(frame, horizon),
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
    if any("_seed_std" in column for column in frame):
        metrics["instability"] = {
            "mean_prediction_std": {column: float(frame[column].mean()) for column in frame if column.endswith("_seed_std")},
            "mean_prediction_variance": {column: float((frame[column] ** 2).mean()) for column in frame if column.endswith("_seed_std")},
            "per_seed": {},
        }
        seeds = sorted({column.rsplit("_seed_", 1)[1] for column in frame if "_seed_" in column and not column.endswith("_seed_std")})
        for seed in seeds:
            seed_frame = frame[[column for column in frame if "_seed_" not in column]].copy()
            for horizon in horizons:
                for suffix in ["pred_return", "direction_prob"]:
                    key = f"{horizon}_{suffix}"
                    seed_frame[key] = frame[f"{key}_seed_{seed}"]
            metrics["instability"]["per_seed"][seed] = _prediction_metrics(seed_frame, horizons)
        metrics["instability"]["metric_variance"] = {
            horizon: {"return_mse": float(np.var([value[horizon]["return"]["mse"] for value in metrics["instability"]["per_seed"].values()])),
                      "direction_accuracy": float(np.var([value[horizon]["direction"]["accuracy"] for value in metrics["instability"]["per_seed"].values()]))}
            for horizon in horizons}

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
        for column in dataset.get("feature_columns", []):
            if column.startswith("diagnostic_noise_"):
                shared[column] = float(dataset["feature_frame"].iloc[max(0, int(row["sample_index"]) - 1)][column])
        for horizon in horizon_names:
            pred_return = float(row[f"{horizon}_pred_return"])
            side = int(np.sign(pred_return))
            direction_prob = float(row[f"{horizon}_direction_prob"])
            confidence = direction_prob if side >= 0 else 1.0 - direction_prob
            horizon_bars = int((label_cfg.get("vertical_barriers", {}) or {}).get(horizon, dataset["horizon_bars"][horizon]))
            chosen_side = int(np.sign(direction_prob - .5) if config.get("candidate_mapping", {}).get("mode") == "probability_sign" else np.sign(pred_return))
            cost_quote = dataset_cost_model(dataset, config).quote({**shared, "horizon": horizon, "horizon_bars": horizon_bars,
                         "side": chosen_side, "position_notional": configured_notional(config)})
            cost_bps = cost_quote["total_cost_bps"]
            expected_edge_bps = abs(pred_return) * 10_000.0 - cost_bps
            mapping = map_forecast(pred_return, direction_prob, float(row.get("realized_vol", .001)),
                                   disagreement, cost_bps, min_edge_bps, config.get("candidate_mapping", {}))
            side, expected_edge_bps, is_candidate = mapping["side"], mapping["expected_edge_bps"], mapping["is_candidate"]
            confidence = direction_prob if side >= 0 else 1. - direction_prob
            out = dict(shared)
            out.update(
                {
                    **mapping, **cost_quote,
                    "horizon": horizon,
                    "horizon_bars": horizon_bars,
                    "position_notional": float(configured_notional(config)),
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
        cost_model=dataset_cost_model(dataset, config),
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
    columns.extend(column for column in frame if column.startswith("diagnostic_noise_"))
    return [column for column in columns if column in frame.columns]


def _fit_meta_core(candidates: pd.DataFrame, config: Mapping[str, Any], horizons: Sequence[str], device: torch.device, dataset=None) -> Any:
    feature_columns = _meta_feature_columns(candidates, horizons)
    train = candidates[(candidates["is_candidate"]) & candidates["meta_label"].notna()].copy()
    if "prediction_role" in train and not train["prediction_role"].isin(["base_oof", "untrained_baseline"]).all():
        raise ValueError("Meta training requires out-of-fold base predictions")
    if train.empty:
        return ConstantMetaModel(0.0, feature_columns, "empty_labeled_candidates")
    meta_cfg = config.get("model", {}).get("meta_labeler", {})
    kind = meta_cfg.get("type", "mlp")
    if kind not in {"constant", "logistic", "tree", "mlp"}:
        raise ValueError(f"Unknown meta-labeler type: {kind}")
    training = config.get("training", {})
    stop_cfg = training.get("early_stopping", {}) or {}
    validation = train.iloc[:0]
    stop_audit = {"status": "disabled"}
    if kind == "mlp" and stop_cfg.get("enabled", False):
        train = train.reset_index(drop=True)
        fit, heldout, stop_audit = training_holdout(train, np.arange(len(train)), config, stop_cfg, dataset=dataset)
        validation, train = train.iloc[heldout].copy(), train.iloc[fit].copy()
    train = _apply_meta_sample_weights(train, config)
    y = train["meta_label"].astype(float).to_numpy()
    if "sample_weight" in train.columns:
        raw_weights = pd.to_numeric(train["sample_weight"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        raw_weights = np.where(raw_weights > 0.0, raw_weights, 0.0)
    else:
        raw_weights = np.ones(len(train), dtype=float)
    if not np.any(raw_weights > 0.0):
        raw_weights = np.ones(len(train), dtype=float)
    train_weights = raw_weights / max(float(raw_weights.mean()), 1e-12)
    if kind == "constant" or len(np.unique(y)) < 2:
        return ConstantMetaModel(float(np.average(y, weights=raw_weights)), feature_columns,
                                 "configured_constant" if kind == "constant" else "single_class_training_labels")
    X = train.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    if kind in {"logistic", "tree"}:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        seed = int(config.get("experiment", {}).get("seed", 42))
        estimator = (LogisticRegression(C=float(meta_cfg.get("C", 1.)), max_iter=1000, random_state=seed)
                     if kind == "logistic" else
                     RandomForestClassifier(n_estimators=int(meta_cfg.get("n_estimators", 50)),
                                            max_depth=int(meta_cfg.get("max_depth", 4)), random_state=seed))
        estimator.fit(X, y, sample_weight=train_weights)
        return SklearnMetaModel(estimator, mean, std, feature_columns, kind)
    torch.manual_seed(int(config.get("experiment", {}).get("seed", torch.initial_seed())) % (2**32))
    model = MetaLabelMLP(
        input_size=X.shape[1],
        hidden_size=int(meta_cfg.get("hidden_size", 16)),
        num_layers=int(meta_cfg.get("num_layers", 1)),
        dropout=float(meta_cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(training.get("learning_rate", 0.001)),
                           weight_decay=float(training.get("weight_decay", 0.)))
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
    stopper = EarlyStopper(stop_cfg.get("patience", 3), stop_cfg.get("min_delta", 0.))
    validation_losses = []
    for epoch in range(int(training.get("epochs", 1))):
        model.train()
        for batch_X, batch_y, batch_w in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad()
            loss = (criterion(model(batch_X), batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()
        if len(validation):
            model.eval()
            vx = validation.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy(dtype=float)
            with torch.no_grad():
                vl = criterion(model(torch.tensor((vx - mean) / std, dtype=torch.float32, device=device)),
                               torch.tensor(validation["meta_label"].to_numpy(), dtype=torch.float32, device=device)).mean()
            validation_losses.append(float(vl.cpu()))
            if stopper.update(validation_losses[-1], epoch + 1, model):
                break
    stopper.restore(model)
    wrapped = TorchMetaModel(model, mean, std, feature_columns, device)
    wrapped.training_history = {"early_stopping": {**stop_audit, "selected_epoch": stopper.best_epoch,
                                "validation_losses": validation_losses, "epochs_run": epoch + 1}}
    return wrapped


def fit_meta_labeler(candidates, config, horizons, device, dataset=None):
    """Fit, calibrate, and select using disjoint purged training-only populations."""
    if "prediction_role" in candidates and not candidates["prediction_role"].isin(["base_oof", "untrained_baseline"]).all():
        raise ValueError("Meta training requires out-of-fold base predictions")
    meta_cfg = config.get("model", {}).get("meta_labeler", {})
    method = (meta_cfg.get("calibration", {}) or {}).get("method", "none")
    if method not in {"none", "platt", "isotonic"}:
        raise ValueError(f"Unknown calibration method: {method}")
    train = candidates[candidates["is_candidate"] & candidates["meta_label"].notna()].reset_index(drop=True)
    before = {"rows": len(candidates), "labeled_rows": int(candidates["meta_label"].notna().sum()),
              "positive_rate": float(candidates["meta_label"].mean()) if candidates["meta_label"].notna().any() else None}
    balance = {"before_filtering": before, "after_filtering": {"rows": len(train),
               "positive_rate": float(train["meta_label"].mean()) if len(train) else None}}
    selection = meta_cfg.get("selection", {}) or {}
    threshold_cfg = config.get("alpha_backtest", {}).get("meta_threshold_selection", {}) or {}
    selection_rows = train.iloc[:0]
    selection_audit = {"status": "disabled"}
    if len(train) and (selection.get("enabled") or threshold_cfg.get("enabled")):
        fit, holdout, selection_audit = training_holdout(train, np.arange(len(train)), config, selection, dataset=dataset)
        selection_rows, train = train.iloc[holdout].copy(), train.iloc[fit].copy()
    calibration_rows = train.iloc[:0]
    calibration_audit = {"status": "disabled", "method": method}
    if len(train) and method != "none":
        settings = meta_cfg.get("calibration", {}) or {}
        fit, holdout, audit = training_holdout(train, np.arange(len(train)), config, settings, dataset=dataset)
        calibration_rows, train = train.iloc[holdout].copy(), train.iloc[fit].copy()
        calibration_audit = {**audit, "method": method}
    types = selection.get("models", [meta_cfg.get("type", "mlp")]) if selection.get("enabled") else [meta_cfg.get("type", "mlp")]
    if not types:
        raise ValueError("Meta model selection needs at least one model")
    # If selection is unavailable, honor the predeclared default model.
    if not len(selection_rows):
        types = [meta_cfg.get("type", "mlp")]
    models, scores = [], []
    metric = selection.get("metric", "brier_score")
    if metric not in {"brier_score", "log_loss"}:
        raise ValueError("Meta selection metric must be brier_score or log_loss")
    for kind in types:
        candidate_config = deepcopy(config)
        candidate_config.setdefault("model", {}).setdefault("meta_labeler", {})["type"] = kind
        base = _fit_meta_core(train, candidate_config, horizons, device, dataset)
        cal, cal_audit = None, dict(calibration_audit)
        if method != "none" and len(calibration_rows):
            truth = calibration_rows["meta_label"].to_numpy(dtype=float)
            if base.kind == "constant" or len(np.unique(truth)) < 2:
                cal_audit.update(status="unavailable", reason="constant model or single-class calibration labels")
            else:
                weights = _apply_meta_sample_weights(calibration_rows, config)["sample_weight"].to_numpy(dtype=float)
                cal = fit_calibrator(method, base.predict_proba(calibration_rows), truth, weights / max(weights.mean(), 1e-12))
                cal_audit.update(status="completed", reason=None)
        bundle = MetaModelBundle(base, cal, method, {"calibration": cal_audit, "class_balance": balance})
        if len(selection_rows):
            truth, probability = selection_rows["meta_label"].to_numpy(dtype=float), bundle.predict_proba(selection_rows)
            value = (float(np.mean((truth-probability)**2)) if metric == "brier_score" else
                     _weighted_binary_classification_summary(truth, probability, np.ones(len(truth)))["log_loss"])
        else:
            value = None
        models.append(bundle)
        scores.append({"model_type": kind, "score": value, "actual_kind": base.kind})
    winner = int(np.argmin([row["score"] for row in scores])) if len(selection_rows) else 0
    model = models[winner]
    model.audit["model_selection"] = {**selection_audit, "metric": metric, "scores": scores, "selected_type": types[winner]}
    model.audit["threshold_selection"] = {"status": "disabled", "source": "configured_fixed_threshold"}
    if threshold_cfg.get("enabled"):
        if len(selection_rows):
            scored = add_meta_probabilities(selection_rows, model)
            scored["prediction_role"] = "meta_training_holdout"
            rows = meta_threshold_table(scored, config.get("alpha_backtest", {}).get("meta_threshold_grid"), config=config)
            objective = threshold_cfg.get("metric", "f1")
            if objective not in {"f1", "validation_net_pnl"}:
                raise ValueError("Threshold metric must be f1 or validation_net_pnl")
            viable = [row for row in rows if row["trade_count"] >= int(threshold_cfg.get("min_trades", 1))]
            chosen = max(viable, key=lambda row: (row[objective], row["threshold"])) if viable else None
            model.decision_threshold = chosen["threshold"] if chosen else None
            model.audit["threshold_selection"] = {"status": "completed" if chosen else "unavailable",
                "reason": None if chosen else "no threshold meets minimum trade count", "metric": objective,
                "source": "purged_meta_training_holdout_only", "table": rows, "split": selection_audit}
        else:
            model.audit["threshold_selection"] = {**selection_audit, "source": "configured_fixed_threshold_fallback"}
    return model


def add_meta_probabilities(candidates: pd.DataFrame, meta_model: Any) -> pd.DataFrame:
    out = candidates.copy()
    out["meta_prob"] = meta_model.predict_proba(out)
    threshold = getattr(meta_model, "decision_threshold", None)
    out["meta_decision_threshold"] = np.nan if threshold is None else threshold
    manifest = meta_model.manifest()
    out["meta_threshold_source"] = manifest.get("threshold_selection", {}).get("source", "configured_fixed_threshold")
    out["meta_model_kind"] = manifest["kind"]
    calibration = manifest.get("calibration", {})
    out["meta_calibration_method"] = calibration.get("method", "none") if calibration.get("status") == "completed" else "none"
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


def run_alpha_backtest(candidates: pd.DataFrame, config: Mapping[str, Any], *, cost_model=None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if "cost_config_hash" in candidates:
        expected_cost_hash = hashlib.sha256(json.dumps(config.get("costs", {}), sort_keys=True).encode()).hexdigest()
        if not candidates.cost_config_hash.eq(expected_cost_hash).all():
            raise ValueError("Candidate labels and backtest use different cost configurations; regenerate labels")
    if cost_model is not None:
        cost_model.validate_records(candidates)
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
        "threshold_mode": "per_model_training_holdout" if candidates.get("meta_decision_threshold", pd.Series(dtype=float)).notna().any() else "configured_fixed",
        "selected_thresholds": sorted(candidates.get("meta_decision_threshold", pd.Series(dtype=float)).dropna().unique().tolist()),
        "edge_threshold_bps": edge_threshold,
        "edge_safety_margin_bps": safety_margin,
        "effective_edge_threshold_bps": effective_edge_threshold,
        "max_trades_per_day": alpha.get("max_trades_per_day"),
        "cooldown_bars": int(alpha.get("cooldown_bars", 0) or 0),
        "cost_model_mode": config.get("costs", {}).get("mode", "static"),
        "simulation_type": "dynamic_cost_simulation" if config.get("costs", {}).get("mode") == "dynamic" else "static_cost_simulation",
        "execution_mode": config.get("costs", {}).get("execution", {}).get("mode", "taker"),
        "execution_diagnostics": {"candidate_rows": len(candidates), "missed_fills": int(candidates.get("filled", pd.Series(True, index=candidates.index)).eq(False).sum())},
    }

    eligible = candidates[
        (candidates["is_candidate"])
        & (candidates["meta_prob"] >= pd.to_numeric(candidates.get("meta_decision_threshold", pd.Series(meta_threshold, index=candidates.index)), errors="coerce").fillna(meta_threshold))
        & (candidates["expected_edge_bps"] >= effective_edge_threshold)
    ].copy()
    if "label_status" in eligible:
        eligible = eligible[eligible["label_status"] == "valid"].copy()
    if "meta_evaluation_status" in eligible:
        eligible = eligible[eligible["meta_evaluation_status"] == "held_out"].copy()
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
        fee_cost = notional * float(row.get("realized_total_cost_bps", row["total_cost_bps"])) * float(row.get("fill_factor", 1.)) / 10_000.0
        gross_pnl = notional * gross_return
        net_pnl = notional * net_return
        trades.append(
            {
                "timestamp": row["timestamp"],
                "pnl_timestamp": row["exit_completed_at"] if pd.notna(row.get("exit_completed_at")) else row["timestamp"],
                "horizon": row["horizon"],
                "side": int(row["side"]),
                "notional": notional * float(row.get("fill_factor", 1.)),
                "requested_notional": notional,
                "entry_price": float(row.get("entry_price", row["close"])),
                **{key: row.get(key) for key in ["entry_idx", "entry_timestamp", "t1_idx", "t1_timestamp", "entry_delay_bars", "execution_mode", "fill_probability", "fill_policy", "fill_factor", "cost_model_id", "cost_model_mode", "simulation_type", "spread_source", "funding_source"]},
                **{name + "_cost": notional * float(row.get("realized_" + name + "_bps", row.get(name + "_bps", 0.))) * float(row.get("fill_factor", 1.)) / 10000. for name in ["fee", "spread", "slippage", "funding"]},
                "gross_return": gross_return,
                "net_return": net_return,
                "gross_pnl": gross_pnl,
                "fees": fee_cost,
                "net_pnl": net_pnl,
                "meta_prob": float(row["meta_prob"]),
                "meta_threshold": float(row["meta_decision_threshold"]) if pd.notna(row.get("meta_decision_threshold")) else meta_threshold,
                "meta_threshold_source": row.get("meta_threshold_source", "configured_fixed_threshold"),
                "meta_model_kind": row.get("meta_model_kind"),
                "meta_calibration_method": row.get("meta_calibration_method", "none"),
                "expected_edge_bps": float(row["expected_edge_bps"]),
                "exit_reason": row.get("exit_reason", "vertical"),
                "label_holding_bars": row.get("label_holding_bars", 0),
                "label_span_start_idx": row.get("label_span_start_idx", np.nan),
                "label_span_end_idx": row.get("label_span_end_idx", np.nan),
                "average_uniqueness": row.get("average_uniqueness", np.nan),
            }
        )
    trades_df = pd.DataFrame(trades)
    total_rows = max(1, int(candidates["timestamp"].nunique()) if "timestamp" in candidates else len(candidates))
    candidate_count = int(candidates["is_candidate"].sum()) if "is_candidate" in candidates else 0
    granularity = str((config.get("data") or {}).get("granularity", "5m"))
    bar_type = config.get("data", {}).get("bar_type") or (config.get("bars", {}) or {}).get("type", "time")
    if bar_type != "time":
        granularity = f"activity:{bar_type}"

    def _statistics(net_pnl: float, turnover: float) -> Dict[str, Any]:
        holding = trades_df["label_holding_bars"] if "label_holding_bars" in trades_df.columns else []
        pnl_series = period_pnl_series(candidates, trades_df)
        result = summarize_single_path(
            pnl_series.to_numpy(dtype=float),
            initial_capital=initial_capital,
            net_pnl=net_pnl,
            turnover=turnover,
            holding_bars=holding,
            trades=trades_df,
            granularity=granularity,
        )
        outflows = sum(float(trades_df[name + "_cost"].clip(lower=0).sum()) for name in ["fee", "spread", "slippage", "funding"] if name + "_cost" in trades_df)
        result["return_on_execution_costs"] = {"available": outflows > 0, "value": net_pnl / outflows if outflows > 0 else "unavailable",
                                              "reason": None if outflows > 0 else "no positive execution cost outflows", "cost_outflows": outflows}
        result["period_timestamps"] = [value.isoformat() for value in pnl_series.index]
        result["pnl_clock"] = "realized_exit_completed_bar" if "pnl_timestamp" in trades_df else "legacy_signal_timestamp"
        return result

    if trades_df.empty:
        metrics = _empty_alpha_metrics(candidate_count, len(candidates), policy)
        metrics["statistics"] = _statistics(0.0, 0.0)
        return metrics, pd.DataFrame(columns=["timestamp", "horizon", "side", "notional", "net_pnl"])

    trades_df["cumulative_net_pnl"] = trades_df["net_pnl"].cumsum()
    equity = initial_capital + trades_df["cumulative_net_pnl"]
    drawdown = equity - equity.cummax().clip(lower=initial_capital)
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
        "cost_components": {name: float(trades_df[name + "_cost"].sum()) for name in ["fee", "spread", "slippage", "funding"]},
        "pnl_per_turnover": float(trades_df.net_pnl.sum() / trades_df.notional.sum()) if trades_df.notional.sum() > 0 else None,
        "break_even_additional_slippage_bps": float(trades_df.net_pnl.sum() / trades_df.notional.sum() * 10000.) if trades_df.notional.sum() > 0 else None,
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
        "statistics": _statistics(float(trades_df["net_pnl"].sum()), float(trades_df["notional"].sum())),
    }
    return metrics, trades_df


def probability_bucket_table(candidates: pd.DataFrame, bins: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    bins = list(bins or np.linspace(0.0, 1.0, 11))
    frame = candidates[candidates["is_candidate"]].copy() if "is_candidate" in candidates else candidates.copy()
    if "meta_evaluation_status" in frame:
        frame = frame[frame["meta_evaluation_status"] == "held_out"]
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


def meta_threshold_table(candidates: pd.DataFrame, thresholds: Optional[Sequence[float]] = None, *, config=None) -> List[Dict[str, Any]]:
    if "prediction_role" in candidates and candidates["prediction_role"].eq("final_test").any():
        raise ValueError("Threshold sweeps cannot use final-test outcomes")
    thresholds = list(thresholds or [0.50, 0.55, 0.60, 0.65, 0.70, 0.80])
    if any(not 0 <= float(value) <= 1 for value in thresholds):
        raise ValueError("Meta thresholds must be within [0, 1]")
    if candidates.empty or "meta_prob" not in candidates or "meta_label" not in candidates:
        return []
    base = candidates[candidates["is_candidate"]].copy() if "is_candidate" in candidates else candidates.copy()
    if "meta_evaluation_status" in base:
        base = base[base["meta_evaluation_status"] == "held_out"]
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
        precision, recall = tp / max(len(selected), 1), tp / max(positives, 1.)
        pnl, trade_count = None, len(selected)
        if config is not None:
            sweep_config = deepcopy(config)
            sweep_config.setdefault("alpha_backtest", {})["meta_threshold"] = float(threshold)
            fixed = labeled.drop(columns=["meta_decision_threshold"], errors="ignore")
            fixed["meta_threshold_source"] = "validation_diagnostic_sweep"
            simulated, _ = run_alpha_backtest(fixed, sweep_config)
            pnl, trade_count = simulated["net_pnl"], int(simulated["trades"])
        cal = calibration_summary(selected["meta_label"], selected["meta_prob"])
        rows.append(
            {
                "threshold": float(threshold),
                "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
                "trade_count": trade_count, "validation_net_pnl": pnl,
                "calibration_error": cal["ece"],
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
    final_test = "prediction_role" in candidates and candidates["prediction_role"].eq("final_test").any()
    return {
        "labels": label_diagnostics(candidates, config),
        "probability_buckets": probability_bucket_table(candidates),
        "predicted_edge_buckets": predicted_edge_bucket_table(candidates),
        "meta_label_thresholds": [] if final_test else meta_threshold_table(candidates, thresholds, config=config),
        "threshold_sweep_status": "not_run_on_final_test" if final_test else "validation_only",
    }


def label_diagnostics(candidates: pd.DataFrame, config: Mapping[str, Any]) -> Dict[str, Any]:
    labeled = candidates[candidates["is_candidate"] & candidates["meta_label"].notna()].copy()
    evaluated = labeled[np.isfinite(pd.to_numeric(labeled.get("meta_prob", pd.Series(np.nan, index=labeled.index)), errors="coerce"))]
    if "meta_evaluation_status" in evaluated:
        evaluated = evaluated[evaluated["meta_evaluation_status"] == "held_out"]
    classification = None
    if len(evaluated):
        truth = evaluated["meta_label"].to_numpy(dtype=float)
        probability = evaluated["meta_prob"].to_numpy(dtype=float)
        classification = _weighted_binary_classification_summary(truth, probability, np.ones(len(evaluated)))
        classification["brier_score"] = float(np.mean((truth - probability) ** 2))
        buckets = np.minimum((np.clip(probability, 0, 1) * 10).astype(int), 9)
        classification["calibration_error"] = float(sum(
            (buckets == bucket).mean() * abs(truth[buckets == bucket].mean() - probability[buckets == bucket].mean())
            for bucket in np.unique(buckets)
        ))
    return {
        "candidate_rows": int(candidates["is_candidate"].sum()),
        "labeled_rows": int(len(labeled)), "evaluated_rows": int(len(evaluated)),
        "positive_rate": float(labeled["meta_label"].mean()) if len(labeled) else None,
        "ambiguous_rows": int(candidates.get("ambiguity_flag", pd.Series(False, index=candidates.index)).sum()),
        "exclusion_reasons": candidates.get("label_exclusion_reason", pd.Series(dtype=object)).value_counts().to_dict(),
        "classification": classification,
        "geometry": dict((config.get("labels", {}) or {}).get("meta_triple_barrier", {})),
        "edge_threshold_coverage": [
            {"threshold_bps": threshold, "eligible_rows": int((candidates["expected_edge_bps"] >= threshold).sum()),
             "coverage": float((candidates["expected_edge_bps"] >= threshold).mean()) if len(candidates) else 0.0}
            for threshold in [0.0, 5.0, 10.0, 25.0, 50.0]
        ],
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
        "method": splits.get("method", "purged_walk_forward"),
        "path_type": splits.get("path_type", "walk_forward"),
        "final_boundary_audit": splits.get("final_boundary_audit"),
        "span_guard": splits.get("span_guard", True),
        "data_coverage": splits.get("data_coverage", {}),
        "purge_bars": int(splits["purge_bars"]),
        "embargo_bars": int(splits["embargo_bars"]),
        "development": describe(splits["development"]),
        "gap": describe(splits["gap"]),
        "test": describe(splits["test"]),
        "folds": [
            {"fold": int(fold["fold"]), "train": describe(fold["train"]), "validation": describe(fold["validation"]), "audit": fold.get("audit")}
            for fold in splits["folds"]
        ],
    }


def _write_stage0(dataset: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    stage_dir = run_dir / "stage0_features"
    stage_dir.mkdir(parents=True, exist_ok=True)
    samples_path = stage_dir / "samples.parquet"
    bars_path = stage_dir / "bars.parquet"
    dataset["price_path"].to_parquet(bars_path, index=False)
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
        "bars_path": bars_path,
    }
    _write_json(stage_dir / "manifest.json", manifest)
    return manifest


def _run_one_trial(
    spec: Mapping[str, Any],
    run_dir: Path,
    *,
    smoke: bool,
    device: torch.device,
    test_contexts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    trial_started = time.perf_counter()
    config = dict(spec["config"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if smoke:
        config = apply_smoke_overrides(config)
    trial_id = str(spec["trial_id"])
    trial_dir = run_dir / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config.get("experiment", {}).get("seed", 42)) + int(spec["trial_index"]))
    np.random.seed(int(config.get("experiment", {}).get("seed", 42)) + int(spec["trial_index"]))

    dataset = build_multi_horizon_lighter_dataset(config, smoke=smoke)
    splits = build_validation_splits(dataset, config)
    split_manifest = _split_manifest(dataset, splits)
    split_manifest_path = trial_dir / "split_manifest.json"
    _write_json(split_manifest_path, split_manifest)
    split_manifest_hash = _stable_hash(split_manifest)

    oof_frames: List[pd.DataFrame] = []
    meta_frames: List[pd.DataFrame] = []
    fold_manifests: List[Dict[str, Any]] = []
    batch_size = int(config.get("training", {}).get("prediction_batch_size", 512))
    importance_folds = []
    for fold in splits["folds"]:
        fold_started = time.perf_counter()
        model, history = train_base_model(dataset, fold["train"], config, device)
        fold_pred = predict_base_model(model, dataset, fold["validation"], device, batch_size=batch_size)
        fold_pred["fold"] = int(fold["fold"])
        fold_pred["prediction_role"] = "base_oof"
        oof_frames.append(fold_pred)
        meta_candidates, meta_audit, fold_meta_model = evaluate_meta_fold(fold_pred, dataset, fold, config, device, return_model=True)
        if config.get("feature_importance", {}).get("enabled", False) and len(importance_folds) < int(config["feature_importance"].get("max_folds", len(splits["folds"]))):
            from src.evaluation.feature_importance import fold_importance
            importance_folds.append(fold_importance(model, fold_meta_model, dataset, fold, meta_candidates, config, device))
        meta_frames.append(meta_candidates)
        path_metrics, _ = run_alpha_backtest(meta_candidates, config, cost_model=dataset_cost_model(dataset, config))
        fold_manifests.append(
            {
                "fold": int(fold["fold"]),
                "elapsed_seconds": time.perf_counter() - fold_started,
                "train_samples": int(len(fold["train"])),
                "validation_samples": int(len(fold["validation"])),
                "history": history,
                "meta_validation": meta_audit,
                "simulated_validation": path_metrics,
                "metrics": _prediction_metrics(fold_pred, dataset["horizon_names"]),
            }
        )

    oof_predictions = collapse_scenario_predictions(pd.concat(oof_frames, ignore_index=True).sort_values("timestamp"))
    final_model, final_history = train_base_model(dataset, splits["development"], config, device)

    # Only validation (out-of-fold) evidence is produced here. The final test
    # split is scored later, and only for trials selected on validation.
    oof_candidates = add_meta_labels(candidate_signals(oof_predictions, dataset, config), dataset, config)
    meta_sample_weight_diagnostics = {
        "validation": oof_candidates.attrs.get("sample_weight_diagnostics", {}),
    }
    meta_model, final_meta_audit = fit_outer_meta_model(dataset, splits["development"], config, device)
    oof_candidates = collapse_scenario_predictions(pd.concat(meta_frames, ignore_index=True), candidates=True)

    validation_metrics, validation_trades = run_alpha_backtest(oof_candidates, config, cost_model=dataset_cost_model(dataset, config))
    diagnostics = {"validation": alpha_diagnostics(oof_candidates, config)}
    from src.experiments.forecast_benchmark import forecast_metrics, zero_return_predictions, momentum_predictions
    validation_metrics["forecast"] = forecast_metrics(oof_predictions, dataset, config)
    validation_indices = dataset["samples"].reset_index().set_index("sample_index").loc[oof_predictions["sample_index"], "index"].to_numpy(dtype=int, copy=True)
    validation_metrics["forecast_baselines"] = {
        "zero_return": forecast_metrics(zero_return_predictions(dataset, validation_indices), dataset, config),
        "momentum": forecast_metrics(momentum_predictions(dataset, validation_indices), dataset, config),
    }
    validation_metrics["candidate_mapping"] = {"settings": config.get("candidate_mapping", {"mode": "expected_edge"}),
                                                "coverage": float(oof_candidates["is_candidate"].mean())}

    validation_metrics["instability"] = validation_metrics["forecast"].get("instability", {"status": "single_seed"})
    validation_metrics["meta_classification"] = diagnostics["validation"]["labels"]["classification"]
    validation_metrics["selection"] = validation_selection_scores(oof_candidates, validation_metrics, config)
    path_values = [item["simulated_validation"]["net_pnl"] for item in fold_manifests if item["meta_validation"]["status"] == "completed"]
    validation_metrics["path_dispersion"] = {
        "path_type": splits.get("path_type"), "scenario_count": len(fold_manifests), "available_scenario_count": len(path_values),
        "net_pnl_quantiles": {str(q): float(np.quantile(path_values, q)) for q in [.1, .5, .9]} if path_values else None,
        "scenarios": [{"fold": item["fold"], "metrics": item["simulated_validation"], "available": item["meta_validation"]["status"] == "completed"} for item in fold_manifests],
        "stitched_cpcv": False,
    }

    from src.evaluation.feature_importance import summarize_importance, write_importance
    from src.evaluation.cost_stress import cost_stress_report
    importance = summarize_importance(importance_folds)
    importance_paths = write_importance(importance, trial_dir, _write_json) if importance_folds else {}
    cost_stress = cost_stress_report(oof_candidates, config)
    _write_json(trial_dir / "cost_stress_validation.json", cost_stress)
    model_path = trial_dir / f"base_multi_horizon_{config.get('model', {}).get('base', {}).get('type', 'lstm')}.pt"
    torch.save(final_model.state_dict(), model_path)
    checkpoint_id = _sha256_path(model_path)
    paths = {
        **importance_paths,
        "cost_stress": trial_dir / "cost_stress_validation.json",
        "oof_predictions": trial_dir / "predictions_oof.parquet",
        "oof_candidates": trial_dir / "meta_candidates_oof.parquet",
        "validation_trades": trial_dir / "alpha_trades_validation.parquet",
        "diagnostics": trial_dir / "alpha_diagnostics.json",
        "forecast_diagnostics": trial_dir / "forecast_diagnostics.json",
        "threshold_sweep": trial_dir / "threshold_sweep_validation.parquet",
        "meta_scenarios": trial_dir / "meta_candidates_scenarios.parquet",
        "meta_validation": trial_dir / "meta_validation.json",
    }
    oof_predictions.to_parquet(paths["oof_predictions"], index=False)
    oof_candidates.to_parquet(paths["oof_candidates"], index=False)
    pd.concat(meta_frames, ignore_index=True).to_parquet(paths["meta_scenarios"], index=False)
    _write_json(paths["meta_validation"], {"folds": fold_manifests})
    validation_trades.to_parquet(paths["validation_trades"], index=False)
    pd.DataFrame(diagnostics["validation"]["meta_label_thresholds"]).to_parquet(paths["threshold_sweep"], index=False)
    _write_json(paths["diagnostics"], diagnostics)
    _write_json(paths["forecast_diagnostics"], {"forecast": validation_metrics["forecast"], "baselines": validation_metrics["forecast_baselines"]})

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
        "final_test_evaluation": None,
        "test_evaluation": {"status": "pending_selection", "reason": "final test is scored only after validation-based selection"},
        "folds": fold_manifests,
        "final_training_history": final_history,
        "feature_importance": importance, "cost_stress": cost_stress,
        "runtime": {"elapsed_seconds": time.perf_counter() - trial_started,
                    "device": str(device), "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None},
        "meta_labeler": meta_model.manifest(),
        "final_meta_training": final_meta_audit,
        "candidate_mapping": config.get("candidate_mapping", {"mode": "expected_edge"}),
        "cost_model": dataset_cost_model(dataset, config).manifest(),
        "meta_sample_weight_diagnostics": meta_sample_weight_diagnostics,
        "model_path": model_path,
        "checkpoint_id": checkpoint_id,
        "artifact_paths": paths,
        "metrics": {"validation": validation_metrics, "test": {}},
        "diagnostics": diagnostics,
        "diagnostics_path": paths["diagnostics"],
        "manifest_path": trial_dir / "manifest.json",
    }
    _write_json(trial_dir / "manifest.json", manifest)
    if test_contexts is not None:
        _move_meta_model(meta_model, torch.device("cpu"))
        test_contexts[trial_id] = {
            "final_model": final_model.to("cpu"),
            "meta_model": meta_model,
            "dataset": dataset,
            "splits": splits,
            "batch_size": batch_size,
            "config": config,
        }
    return manifest


def _move_meta_model(meta_model: Any, device: torch.device) -> None:
    if isinstance(meta_model, MetaModelBundle):
        _move_meta_model(meta_model.base, device)
    if isinstance(meta_model, TorchMetaModel):
        meta_model.model.to(device)
        meta_model.device = device


def _score_trial_on_test(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    run_dir: Path,
    smoke: bool,
    device: torch.device,
    selection_roles: Sequence[str],
) -> Dict[str, Any]:
    """Backtest the final test split for one validation-selected trial.

    The reuse ledger is updated before any test prediction is computed.
    """

    config = context["config"]
    dataset = context["dataset"]
    splits = context["splits"]
    trial_id = str(manifest["trial_id"])
    final_test_evaluation = _record_final_test_evaluation(
        config,
        run_dir,
        trial_id=trial_id,
        split_hash=str(manifest["split_manifest_hash"]),
        smoke=smoke,
        scope="meta_label_trial",
    )
    final_model = context["final_model"].to(device)
    meta_model = context["meta_model"]
    _move_meta_model(meta_model, device)
    test_predictions = predict_base_model(
        final_model,
        dataset,
        splits["test"],
        device,
        batch_size=int(context["batch_size"]),
    )
    test_predictions["prediction_role"] = "final_test"
    test_candidates = add_meta_probabilities(
        add_meta_labels(candidate_signals(test_predictions, dataset, config), dataset, config),
        meta_model,
    )
    test_metrics, test_trades = run_alpha_backtest(test_candidates, config, cost_model=dataset_cost_model(dataset, config))
    from src.experiments.forecast_benchmark import forecast_metrics
    test_metrics["forecast"] = forecast_metrics(test_predictions, dataset, config)
    test_metrics["instability"] = test_metrics["forecast"].get("instability", {"status": "single_seed"})
    test_metrics["meta_classification"] = label_diagnostics(test_candidates, config)["classification"]
    diagnostics = dict(manifest.get("diagnostics") or {})
    diagnostics["test"] = alpha_diagnostics(test_candidates, config)
    meta_diagnostics = dict(manifest.get("meta_sample_weight_diagnostics") or {})
    meta_diagnostics["test"] = test_candidates.attrs.get("sample_weight_diagnostics", {})

    trial_dir = Path(manifest["manifest_path"]).parent
    paths = dict(manifest.get("artifact_paths") or {})
    paths["forecast_diagnostics_test"] = trial_dir / "forecast_diagnostics_test.json"
    _write_json(paths["forecast_diagnostics_test"], test_metrics["forecast"])
    paths["test_predictions"] = trial_dir / "predictions_test.parquet"
    paths["test_candidates"] = trial_dir / "meta_candidates_test.parquet"
    paths["test_trades"] = trial_dir / "alpha_trades_test.parquet"
    paths["diagnostics"] = trial_dir / "alpha_diagnostics.json"
    test_predictions.to_parquet(paths["test_predictions"], index=False)
    test_candidates.to_parquet(paths["test_candidates"], index=False)
    test_trades.to_parquet(paths["test_trades"], index=False)
    _write_json(paths["diagnostics"], diagnostics)

    updated = dict(manifest)
    metrics = dict(updated.get("metrics") or {})
    metrics["test"] = test_metrics
    updated["metrics"] = metrics
    updated["diagnostics"] = diagnostics
    updated["diagnostics_path"] = paths["diagnostics"]
    updated["artifact_paths"] = paths
    updated["meta_sample_weight_diagnostics"] = meta_diagnostics
    updated["final_test_evaluation"] = final_test_evaluation
    updated["test_evaluation"] = {
        "status": "evaluated",
        "reason": "trial selected on validation evidence",
        "selection_roles": list(selection_roles),
    }
    _write_json(trial_dir / "manifest.json", updated)
    final_model.to("cpu")
    _move_meta_model(meta_model, torch.device("cpu"))
    return updated


def _apply_selected_test_evaluations(
    trial_manifests: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    contexts: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    smoke: bool,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Score only trials chosen by validation roles, and record each one in the ledger."""

    roles = set(final_test_selection_roles(config))
    if selection.get("selection_score_available") is False:
        roles = set()
    updated: List[Dict[str, Any]] = []
    for trial in trial_manifests:
        current = dict(trial)
        if current.get("status") != "completed":
            updated.append(current)
            continue
        trial_roles = _selection_roles_for_trial(current, selection)
        if not (set(trial_roles) & roles):
            current["test_evaluation"] = {
                "status": "disabled" if not roles else "not_selected",
                "reason": "no validation score available" if selection.get("selection_score_available") is False else "final-test scoring disabled by configuration" if not roles else "final test is scored only for trials selected on validation",
                "selection_roles": trial_roles,
            }
            current["final_test_evaluation"] = None
            metrics = dict(current.get("metrics") or {})
            metrics["test"] = {}
            current["metrics"] = metrics
            manifest_path = current.get("manifest_path")
            if manifest_path:
                _write_json(Path(manifest_path), current)
            updated.append(current)
            continue
        context = contexts.get(str(current.get("trial_id")))
        if context is None:
            raise RuntimeError(f"Selected trial {current.get('trial_id')} is missing its test-evaluation context")
        updated.append(
            _score_trial_on_test(
                current,
                context,
                run_dir=run_dir,
                smoke=smoke,
                device=device,
                selection_roles=trial_roles,
            )
        )
    return updated


def _carry_selection_annotations(
    annotated: Optional[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not annotated:
        return None
    match = next((item for item in manifests if item.get("trial_id") == annotated.get("trial_id")), None)
    if match is None:
        return dict(annotated)
    merged = dict(match)
    if "selected_by" in annotated:
        merged["selected_by"] = annotated["selected_by"]
    if "selection_status" in annotated:
        merged["selection_status"] = annotated["selection_status"]
    return merged


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
        "coverage_report": getattr(error, "report", None),
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


def _run_trial_safe(
    spec: Mapping[str, Any],
    run_dir: Path,
    *,
    smoke: bool,
    device: torch.device,
    test_contexts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    trial_cfg = (spec.get("config", {}) or {}).get("trial", {})
    if isinstance(trial_cfg, Mapping) and bool(trial_cfg.get("skip", False)):
        return _skipped_trial_manifest(
            spec,
            run_dir,
            smoke=smoke,
            reason=str(trial_cfg.get("skip_reason", "trial marked skipped by config")),
        )
    try:
        return _run_one_trial(spec, run_dir, smoke=smoke, device=device, test_contexts=test_contexts)
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
    metric = ALIASES.get(str(pipeline.get("selection_metric", "metrics.validation.net_pnl")), str(pipeline.get("selection_metric", "metrics.validation.net_pnl")))
    mode = str(pipeline.get("selection_mode", "max"))
    if mode not in {"min", "max"}:
        raise ValueError("selection_mode must be min or max")
    def key(item):
        value = _metric_value(item, metric)
        return (not math.isfinite(value), -value if mode == "max" else value)
    return sorted(trials, key=key)


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
    direct = _metric_value(trial, "metrics.validation.meta_classification.calibration_error")
    if math.isfinite(direct):
        return direct
    if not validation or not sum(row.get("count", 0) for row in validation.get("probability_buckets", [])):
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
    validate_selection_metric(config)
    pipeline = config.get("pipeline", {}) or {}
    metric = ALIASES.get(str(pipeline.get("selection_metric", "metrics.validation.net_pnl")), str(pipeline.get("selection_metric", "metrics.validation.net_pnl")))
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

    meta_classified = [trial for trial in trials if math.isfinite(_metric_value(trial, "metrics.validation.meta_classification.f1"))]
    classification_best = max(meta_classified, key=lambda trial: _metric_value(trial, "metrics.validation.meta_classification.f1")) if meta_classified else max(trials, key=_mean_validation_direction_accuracy)
    calibration_candidates = [trial for trial in trials if math.isfinite(_trial_calibration_error(trial))]
    calibration_best = min(calibration_candidates, key=_trial_calibration_error) if calibration_candidates else None

    selection = {
        "metric": metric,
        "mode": mode,
        "selection_score_available": math.isfinite(_metric_value(raw_best, metric)),
        "min_validation_trades": min_trades,
        "raw_best_trial_id": raw_best.get("trial_id"),
        "raw_best_validation_trades": raw_best_trades,
        "raw_best_selection_status": raw_best["selection_status"],
        "best_trading_trial_id": best_trading.get("trial_id") if best_trading else None,
        "classification_best_trial_id": classification_best.get("trial_id") if classification_best else None,
        "classification_best_score": (_metric_value(classification_best, "metrics.validation.meta_classification.f1") if meta_classified else _mean_validation_direction_accuracy(classification_best)) if classification_best else None,
        "classification_score_source": "held_out_meta_f1" if meta_classified else "legacy_base_direction_accuracy",
        "calibration_best_trial_id": calibration_best.get("trial_id") if calibration_best else None,
        "calibration_best_ece": _trial_calibration_error(calibration_best) if calibration_best else None,
        "qualified_trial_count": int(len(qualified)),
    }
    winners = {}
    for name, path, maximize in [("f1", "metrics.validation.selection.f1", True),
                                 ("negative_log_loss", "metrics.validation.selection.negative_log_loss", True),
                                 ("calibration_error", "metrics.validation.selection.calibration_error", False),
                                 ("edge_rank", "metrics.validation.selection.edge_rank_correlation", True),
                                 ("net_pnl", "metrics.validation.net_pnl", True),
                                 ("composite", "metrics.validation.selection.composite", True)]:
        available = [row for row in trials if math.isfinite(_metric_value(row, path))]
        winner = (max if maximize else min)(available, key=lambda row: _metric_value(row, path)) if available else None
        winners[name] = {"trial_id": winner.get("trial_id") if winner else None,
                         "score": _metric_value(winner, path) if winner else None,
                         "trade_qualified": _validation_trade_count(winner) >= min_trades if winner else False}
    selection["validation_metric_winners"] = winners
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


def _concentrated_pnl(validation: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    """Flag PnL concentration from the positive-return HHI, not from coverage."""

    statistics = validation.get("statistics") if isinstance(validation, Mapping) else {}
    hhi = (statistics or {}).get("hhi_positive_pnl") if isinstance(statistics, Mapping) else {}
    value = hhi.get("value") if isinstance(hhi, Mapping) else None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    return float(value) >= float(thresholds.get("concentrated_pnl_hhi", 0.5))


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
        "concentrated_pnl": _concentrated_pnl(validation, thresholds),
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
        if isinstance(trial, dict):
            trial["failure_modes"] = failure_modes
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
                "reason": trial.get("reason"),
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


_ROW_LEVEL_SUFFIXES = ("_indices", "_keys")


def _trackio_compact(value: Any) -> Any:
    """Replace row-level ID lists with counts; the local manifest keeps them."""
    if isinstance(value, Mapping):
        compact: Dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, (list, tuple)) and (str(key).endswith(_ROW_LEVEL_SUFFIXES) or key == "draws"):
                compact[f"{key}_count"] = len(item)
            else:
                compact[key] = _trackio_compact(item)
        return compact
    if isinstance(value, (list, tuple)):
        return [_trackio_compact(item) for item in value]
    return value


def _fold_metrics_for_trackio(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    # Per-fold seed variance stays; full per-seed metric copies stay in the manifest.
    metrics = dict(metrics or {})
    instability = metrics.get("instability")
    if isinstance(instability, Mapping):
        metrics["instability"] = {key: value for key, value in instability.items() if key != "per_seed"}
    return metrics


def _training_loss_metrics(history: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    history = history or {}
    if history.get("members"):
        return {f"seed_{seed}": _training_loss_metrics(member)
                for seed, member in zip(history.get("ensemble_seeds", []), history["members"])}
    return {f"epoch_{index + 1}_loss": loss for index, loss in enumerate(history.get("training_losses", []))}


def _log_meta_label_trial_trackio(
    config: Mapping[str, Any],
    *,
    run_id: str,
    trial: Mapping[str, Any],
    selection_status: str,
    raw_best: Mapping[str, Any],
    best_trading: Optional[Mapping[str, Any]],
    selection: Mapping[str, Any],
    trial_count: int,
    smoke: bool,
) -> None:
    trial_id = str(trial.get("trial_id", "unknown"))
    artifacts = {"manifest": trial.get("manifest_path"), "diagnostics": trial.get("diagnostics_path"), "model": trial.get("model_path")}
    artifacts.update(trial.get("artifact_paths", {}) or {})
    artifacts = {key: value for key, value in artifacts.items() if value is not None}
    raw_best_id = raw_best.get("trial_id") if isinstance(raw_best, Mapping) else None
    trading_id = best_trading.get("trial_id") if isinstance(best_trading, Mapping) else None
    is_raw_best = raw_best_id is not None and trial_id == raw_best_id
    is_best_trading = trading_id is not None and trial_id == trading_id
    selection_roles = _selection_roles_for_trial(trial, selection)
    failure_modes = trial.get("failure_modes") or {"flags": {}, "metrics": {}}

    run_config = {
        "stage": "meta_label_mvp_trial",
        "run_id": run_id,
        "trial_id": trial_id,
        "trial_index": trial.get("trial_index"),
        "trial_count": trial_count,
        "checkpoint_id": trial.get("checkpoint_id"),
        "config_hash": (trial.get("run_identity", {}) or {}).get("config_hash"),
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
        "reason": trial.get("reason"),
        "skip_reason": trial.get("skip_reason"),
        "split_manifest_hash": trial.get("split_manifest_hash"),
        "feature_families": ((trial.get("dataset", {}) or {}).get("feature_manifest", {}) or {}).get("families", []),
        "costs": (trial.get("config", {}) or {}).get("costs", {}),
        "alpha_policy": (trial.get("config", {}) or {}).get("alpha_backtest", {}),
        "final_test_evaluation": trial.get("final_test_evaluation"),
        "feature_manifest": (trial.get("dataset", {}) or {}).get("feature_manifest"),
        "bar_manifest": (trial.get("dataset", {}) or {}).get("bar_manifest"),
        "event_diagnostics": (trial.get("dataset", {}) or {}).get("event_diagnostics"),
        "sample_weight_diagnostics": (trial.get("dataset", {}) or {}).get("sample_weight_diagnostics"),
        "bar_clock_diagnostics": (trial.get("dataset", {}) or {}).get("bar_clock_diagnostics"),
        "side_data_manifest": (trial.get("dataset", {}) or {}).get("side_data_manifest"),
        "meta_labeler": _trackio_compact(trial.get("meta_labeler")),
        "training_discipline": _trackio_compact(trial.get("final_training_history")),
        "candidate_mapping": trial.get("candidate_mapping"),
        "cost_model": trial.get("cost_model"),
        "resolved_base_model": trial.get("config", {}).get("model", {}).get("base", {}),
        "resolved_training": trial.get("config", {}).get("training", {}),
        "resolved_labels": trial.get("config", {}).get("labels", {}),
        "feature_importance": trial.get("feature_importance", {}).get("summary", []),
    }
    split_metrics = trial.get("metrics", {}) or {}
    metrics = {
        "alpha": {split: {key: value for key, value in payload.items()
                          if key not in {"forecast", "forecast_baselines", "instability"}}
                  for split, payload in split_metrics.items()},
        # Seed instability is logged once, under model_instability.
        "forecast": {split: {key: value for key, value in (payload.get("forecast", {}) or {}).items() if key != "instability"}
                     for split, payload in split_metrics.items()},
        "forecast_baselines": {split: payload.get("forecast_baselines", {}) for split, payload in split_metrics.items()},
        "model_instability": {split: payload.get("instability", {}) for split, payload in split_metrics.items()},
        "feature_importance": {f"{row['model']}.{row['feature']}": {"negative_log_loss_drop": row["negative_log_loss_drop"], "fold_std": row["fold_std"], "f1_drop": row["f1_drop"]} for row in trial.get("feature_importance", {}).get("summary", [])},
        "cost_stress": {str(row["cost_multiple"]): row for row in trial.get("cost_stress", {}).get("rows", [])},
        "runtime": trial.get("runtime", {}),
        "forecast_oof_by_fold": {
            f"fold_{fold.get('fold')}": _fold_metrics_for_trackio(fold.get("metrics", {}))
            for fold in trial.get("folds", []) or []
        },
        "training_final": _training_loss_metrics(trial.get("final_training_history")),
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
            "failure_modes": failure_modes,
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
        smoke=smoke,
        receipt_path=Path(trial["manifest_path"]).parent / "trackio_receipt.json",
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
        "trial_count": report.get("trial_count"),
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
        "evidence_grade": report.get("evidence_grade"),
        "path_type": report.get("path_type"),
        "pbo": report.get("pbo"),
        "annualisation": best.get("metrics", {}).get("validation", {}).get("statistics", {}).get("annualisation", ANNUALISATION_NOTE),
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
        smoke=smoke,
        receipt_path=Path(report["run_dir"]) / "tracking" / "summary.json",
    )


def _statistic_value(trial: Mapping[str, Any], split: str, name: str) -> Any:
    statistics = (((trial.get("metrics") or {}).get(split) or {}).get("statistics") or {})
    payload = statistics.get(name) or {}
    if isinstance(payload, Mapping):
        return payload.get("value", "unavailable")
    return "unavailable"


def _trial_receipt_path(trial: Mapping[str, Any]) -> Optional[Path]:
    manifest_path = trial.get("manifest_path")
    if not manifest_path:
        return None
    return Path(manifest_path).parent / "trackio_receipt.json"


def _write_trial_manifests(trials: Sequence[Mapping[str, Any]]) -> None:
    for trial in trials:
        manifest_path = trial.get("manifest_path")
        if manifest_path:
            _write_json(Path(manifest_path), trial)


def _stamp_trackio_receipts(
    trials: Sequence[Dict[str, Any]],
    accounting: Dict[str, Any],
    *,
    accounting_path: Path,
    summary_receipt: Optional[Path] = None,
) -> None:
    """Point local trial manifests and the accounting file at Trackio receipt files."""

    by_id = {row.get("trial_id"): row for row in accounting.get("trials", []) if isinstance(row, dict)}
    for trial in trials:
        receipt = _trial_receipt_path(trial)
        if receipt is None:
            continue
        trial["trackio_receipt_path"] = receipt
        row = by_id.get(trial.get("trial_id"))
        if isinstance(row, dict):
            row["trackio_receipt_path"] = receipt
    if summary_receipt is not None:
        accounting["summary_trackio_receipt_path"] = summary_receipt
    _write_trial_manifests(trials)
    _write_json(accounting_path, accounting)


def _log_saved_trial_trackio(
    config: Mapping[str, Any],
    *,
    run_id: str,
    trials: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    smoke: bool,
    raw_best: Optional[Mapping[str, Any]] = None,
    best_trading: Optional[Mapping[str, Any]] = None,
) -> None:
    for trial in trials:
        if raw_best is None:
            selection_status = str(trial.get("status", "failed"))
        else:
            selection_status = _selection_status_for_trial(trial, raw_best, best_trading, selection)
        _log_meta_label_trial_trackio(
            config,
            run_id=run_id,
            trial=trial,
            selection_status=selection_status,
            raw_best=raw_best or {},
            best_trading=best_trading,
            selection=selection,
            trial_count=len(trials),
            smoke=smoke,
        )


def _log_accounting_summary_trackio(
    config: Mapping[str, Any],
    *,
    run_id: str,
    run_dir: Path,
    accounting: Mapping[str, Any],
    smoke: bool,
    status: str,
) -> Path:
    receipt_path = run_dir / "tracking" / "accounting.json"
    log_trackio_run(
        config,
        name=f"meta_label/{run_id}/accounting",
        group="meta_label_mvp",
        run_config={
            "stage": "meta_label_mvp_accounting",
            "run_id": run_id,
            "trial_count": accounting.get("trial_count"),
            "counts": accounting.get("counts", {}),
            "failure_modes": accounting.get("failure_modes", {}),
            "smoke": smoke,
        },
        metrics={
            "trial_accounting": accounting.get("failure_modes", {}),
            "counts": accounting.get("counts", {}),
        },
        status=status,
        smoke=smoke,
        receipt_path=receipt_path,
    )
    return receipt_path


def _raise_after_logging_failed_trials(
    config: Mapping[str, Any],
    *,
    run_id: str,
    run_dir: Path,
    trials: Sequence[Dict[str, Any]],
    smoke: bool,
) -> None:
    """Persist and log the accounting summary, then raise."""

    min_trades = int((config.get("pipeline", {}) or {}).get("min_validation_trades", 1) or 0)
    selection = {"min_validation_trades": min_trades}
    accounting = _trial_accounting(trials, selection, config)
    accounting_path = _write_json(run_dir / "trial_accounting.json", accounting)
    _write_trial_manifests(trials)
    summary_receipt = _log_accounting_summary_trackio(
        config,
        run_id=run_id,
        run_dir=run_dir,
        accounting=accounting,
        smoke=smoke,
        status="failed",
    )
    _log_saved_trial_trackio(config, run_id=run_id, trials=trials, selection=selection, smoke=smoke)
    _stamp_trackio_receipts(trials, accounting, accounting_path=accounting_path, summary_receipt=summary_receipt)
    raise RuntimeError("All v2 trial specs failed; see trial manifests for failure reasons")


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
        "- Test net PnL: `{}`".format((best["metrics"].get("test") or {}).get("net_pnl", "not_evaluated")),
        "- Test trades: `{}`".format((best["metrics"].get("test") or {}).get("trades", "not_evaluated")),
        "- Test coverage: `{}`".format((best["metrics"].get("test") or {}).get("coverage", "not_evaluated")),
        "- Evidence grade: `{}`".format(report.get("evidence_grade", "debugging_only")),
        "- Path type: `{}`".format(report.get("path_type", "single_path")),
        "- Base training: `{}`".format(best.get("final_training_history", {}).get("ensemble_seeds", "single seed")),
        "- Meta model: `{}`".format(best.get("meta_labeler", {}).get("kind", "unavailable")),
        "- Meta calibration: `{}` / `{}`".format(best.get("meta_labeler", {}).get("calibration", {}).get("method", "none"), best.get("meta_labeler", {}).get("calibration", {}).get("status", "unavailable")),
        "- Validation classification/calibration: `{}`".format(best["metrics"]["validation"].get("meta_classification")),
        "- Seed instability: `{}`".format(best["metrics"]["validation"].get("instability", {}).get("metric_variance", "single seed")),
        "- Annualisation: `{}`".format(best["metrics"]["validation"].get("statistics", {}).get("annualisation", ANNUALISATION_NOTE)),
        "- Validation Sharpe (per period): `{}`".format(_statistic_value(best, "validation", "sharpe_per_period")),
        "- Validation PSR: `{}`".format(_statistic_value(best, "validation", "psr")),
        "- Validation DSR: `{}`".format(_statistic_value(best, "validation", "dsr")),
        "- Test Sharpe (per period): `{}`".format(_statistic_value(best, "test", "sharpe_per_period")),
        "- Test DSR: `{}`".format(_statistic_value(best, "test", "dsr")),
        "- PBO: `{}`".format((report.get("pbo") or {}).get("value", "unavailable")),
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


def validation_test_rank_stability(trials):
    pairs = [(trial["trial_id"], trial.get("metrics", {}).get("validation", {}).get("net_pnl"),
              trial.get("metrics", {}).get("test", {}).get("net_pnl")) for trial in trials]
    pairs = [row for row in pairs if row[1] is not None and row[2] is not None]
    result = {"available": False, "spearman": None, "trials": pairs,
              "scope": "validation-selected trials with recorded final-test evaluations only",
              "reason": "fewer than two evaluated trials or constant ranks"}
    if len(pairs) >= 2:
        correlation = pd.Series([row[1] for row in pairs]).corr(pd.Series([row[2] for row in pairs]), method="spearman")
        if np.isfinite(correlation):
            result.update(available=True, spearman=float(correlation), reason=None)
    return result


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

    validate_selection_metric(config_for_search)
    enforce_trackio_policy(config_for_search, smoke=smoke)

    pipeline = config_for_search.get("pipeline", {}) or {}
    run_id = _run_id(str(pipeline.get("run_name", "meta_label_mvp")))
    run_dir = Path(pipeline.get("artifact_root", "artifacts/runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_training_device(device, allow_cpu=allow_cpu)
    search_plan = build_search_plan(config_for_search)
    _write_json(run_dir / "search_plan.json", search_plan)
    trial_specs = expand_grid_search(config_for_search)

    resolved_config_path = _write_yaml(run_dir / "resolved_config.yml", config_for_search)
    environment_path = _write_json(run_dir / "environment.json", _environment_manifest())
    raw_data = _raw_data_manifest(config_for_search)
    artifact_root_path = Path(pipeline.get("artifact_root", "artifacts/runs"))
    raw_data_guard = check_raw_data_registry(
        config_for_search,
        artifact_root_path,
        run_id=run_id,
        raw_manifest=raw_data,
        smoke=smoke,
    )
    run_identity = {
        **_config_identity(config_for_search, resolved_config_path),
        "source_config_path": config_path,
        "run_id": run_id,
        "run_dir": run_dir,
        "research_spec_hash": research_spec_hash(config_for_search),
        "raw_data": raw_data,
        "raw_data_guard": raw_data_guard,
        "git": _git_manifest(),
        "dependencies": _dependency_manifest(),
        "device": _device_manifest(resolved_device),
        "trial_count": len(trial_specs),
        "search_plan_hash": _stable_hash(search_plan),
        "environment_path": environment_path,
    }
    _write_json(run_dir / "run_identity.json", run_identity)
    try:
        stage0_dataset = build_multi_horizon_lighter_dataset(config_for_search, smoke=smoke)
    except DataCoverageError as error:
        _write_json(run_dir / "coverage_failure.json", error.report)
        raise
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

    test_contexts: Dict[str, Dict[str, Any]] = {}
    trial_manifests = [
        _run_trial_safe(spec, run_dir, smoke=smoke, device=resolved_device, test_contexts=test_contexts) for spec in trial_specs
    ]
    completed_trials = [trial for trial in trial_manifests if trial.get("status") == "completed"]
    if not completed_trials:
        _raise_after_logging_failed_trials(
            config_for_search,
            run_id=run_id,
            run_dir=run_dir,
            trials=trial_manifests,
            smoke=smoke,
        )
    best, best_trading, selection = select_trials_with_trade_floor(completed_trials, config_for_search)
    trial_manifests = _apply_selected_test_evaluations(
        trial_manifests,
        selection,
        test_contexts,
        config=config_for_search,
        run_dir=run_dir,
        smoke=smoke,
        device=resolved_device,
    )
    best = _carry_selection_annotations(best, trial_manifests)
    best_trading = _carry_selection_annotations(best_trading, trial_manifests)
    best_path = run_dir / "best_trial_manifest.json"
    _write_json(best_path, best)
    best_trading_path = None
    if best_trading is not None:
        best_trading_path = run_dir / "best_trade_qualified_trial_manifest.json"
        _write_json(best_trading_path, best_trading)

    trial_accounting = _trial_accounting(trial_manifests, selection, config_for_search)
    research_budget = comparison_budget(trial_manifests)
    trial_accounting["research_comparison_budget"] = research_budget
    cross_trial = attach_cross_trial_statistics(trial_manifests, trial_count=research_budget["conservative_comparison_count"])
    test_metrics = (best or {}).get("metrics", {}).get("test", {}) if isinstance(best, Mapping) else {}
    test_dsr = ((test_metrics.get("statistics") or {}).get("dsr") or {})
    final_test_uses = final_test_use_count(trial_manifests)
    evidence_grade = classify_evidence_grade(
        trade_count=float(test_metrics.get("trades") or 0.0),
        dsr_available=bool(test_dsr.get("available")),
        final_test_uses=final_test_uses,
        smoke=smoke,
        min_trades=int(selection.get("min_validation_trades", 0) or 0),
    )
    trial_accounting_path = _write_json(run_dir / "trial_accounting.json", trial_accounting)
    _write_trial_manifests(trial_manifests)

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
        "search_plan": search_plan, "search_plan_path": run_dir / "search_plan.json",
        "research_comparison_budget": research_budget,
        "benchmark": benchmark_summary,
        "trials": trial_manifests,
        "best_trial": best,
        "best_trial_manifest_path": best_path,
        "best_trading_trial": best_trading,
        "best_trading_trial_manifest_path": best_trading_path,
        "selection": selection,
        "trial_accounting": trial_accounting,
        "trial_accounting_path": trial_accounting_path,
        "path_type": "multi_path" if best.get("metrics", {}).get("validation", {}).get("path_dispersion", {}).get("path_type") in {"grouped_scenario", "grouped_scenarios", "purged_kfold"} else "single_path",
        "validation_test_rank_stability": validation_test_rank_stability(trial_manifests),
        "evidence_grade": evidence_grade,
        "pbo": cross_trial["pbo"],
        "annualisation": best.get("metrics", {}).get("validation", {}).get("statistics", {}).get("annualisation", ANNUALISATION_NOTE),
        "status": "success",
        "report_json_path": report_dir / "report.json",
        "report_markdown_path": report_dir / "report.md",
    }
    from src.evaluation.evidence import assess_evidence
    report["final_test_use_count"] = final_test_uses
    report["evidence_assessment"] = assess_evidence(report, config_for_search)
    report["legacy_presence_grade"] = report["evidence_grade"]
    report["evidence_grade"] = report["evidence_assessment"]["grade"]
    _write_json(report["report_json_path"], report)
    Path(report["report_markdown_path"]).write_text(_report_markdown(report), encoding="utf-8")
    _write_json(run_dir / "manifest.json", report)
    _log_saved_trial_trackio(
        config_for_search,
        run_id=run_id,
        trials=trial_manifests,
        selection=selection,
        smoke=smoke,
        raw_best=best,
        best_trading=best_trading,
    )
    _log_meta_label_summary_trackio(config_for_search, run_id=run_id, report=report, smoke=smoke)
    summary_receipt = _log_accounting_summary_trackio(
        config_for_search,
        run_id=run_id,
        run_dir=run_dir,
        accounting=trial_accounting,
        smoke=smoke,
        status="success",
    )
    _stamp_trackio_receipts(
        trial_manifests,
        trial_accounting,
        accounting_path=trial_accounting_path,
        summary_receipt=summary_receipt,
    )
    best = _carry_selection_annotations(best, trial_manifests)
    best_trading = _carry_selection_annotations(best_trading, trial_manifests)
    if best is not None:
        _write_json(best_path, best)
        report["best_trial"] = best
    if best_trading is not None and best_trading_path is not None:
        _write_json(best_trading_path, best_trading)
        report["best_trading_trial"] = best_trading
    report["trial_accounting"] = trial_accounting
    if config_for_search.get("reporting", {}).get("dashboard", True):
        from src.experiments.dashboard import build_dashboard
        report["dashboard_path"] = build_dashboard(report)
    _write_json(report["report_json_path"], report)
    _write_json(run_dir / "manifest.json", report)
    return _json_ready(report)
