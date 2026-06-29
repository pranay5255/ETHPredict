
"""Staged Lighter experiment orchestration with durable artifacts.

This runner turns the Lighter experiment flow into explicit stages:

1. Stage 0 materializes and records the dataset/feature contract.
2. Stage 1 trains one or more hyperparameter trials and selects a best model.
3. Stage 2 consumes the best Stage 1 prediction artifact for strategy backtests.
4. Stage 3 writes machine-readable and human-readable reports.

The stage boundaries are intentionally file-backed so longer searches can be run
by changing config values without losing the ability to inspect or reproduce a
single trial.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import re
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.features_all import DEFAULT_GRANULARITY, DataPreprocessor, bars_for_duration
from src.experiments.lighter_compare import _deep_update, _training_config, build_lighter_dataset, load_experiment_config
from src.market_maker.glft import GLFTParams, GLFTQuoteCalculator
from src.market_maker.inventory import InventoryBook
from src.simulator.backtest import BacktestParams, GLFTBacktester
from src.training.devices import resolve_training_device
from src.training.trainer import HierarchicalPredictor, compute_metrics, hierarchical_training_pipeline


DEFAULT_STAGED_CONFIG: Dict[str, Any] = {
    "artifact_root": "artifacts/runs",
    "run_name": "staged_lighter_trial",
    "target": "next_hour_return",
    "selection_metric": "metrics.validation.prediction.mae",
    "selection_mode": "min",
    "save_stage0_tensors": False,
    "split": {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "purge_bars": 0,
        "embargo_bars": 0,
    },
    "stage1": {
        "max_trials": None,
        "trials": [],
        "search_space": {},
        "prediction_batch_size": 512,
    },
    "stage2": {
        "max_rows": None,
        "prediction_clip": 0.05,
        "selection_metric": "metrics.net_pnl",
        "selection_mode": "max",
        "strategy": {},
        "strategies": [],
        "strategy_search": {},
    },
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "run"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Size):
        return list(value)
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


def _sha256(path: Path) -> str:
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
        "git_head": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": _command_output(["git", "status", "--short"]),
    }


def _staged_config(config: Dict[str, Any], *, smoke: bool) -> Dict[str, Any]:
    staged = _deep_update(DEFAULT_STAGED_CONFIG, config.get("staged_trial", {}) or {})
    if smoke:
        staged = _deep_update(staged, staged.get("smoke", {}) or {})
    return staged


def _run_id(run_name: str) -> str:
    return f"{_slug(run_name)}_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"


def _metric_value(payload: Mapping[str, Any], path: str) -> float:
    candidates = [path]
    if not path.startswith("metrics."):
        candidates.append(f"metrics.{path}")
    for candidate in candidates:
        current: Any = payload
        ok = True
        for part in candidate.split("."):
            if not isinstance(current, Mapping) or part not in current:
                ok = False
                break
            current = current[part]
        if ok:
            return _safe_float(current, default=math.inf)
    raise KeyError(f"Metric path not found: {path}")


def _rank_payloads(payloads: Sequence[Mapping[str, Any]], metric: str, mode: str) -> List[Mapping[str, Any]]:
    reverse = mode == "max"
    return sorted(payloads, key=lambda item: _metric_value(item, metric), reverse=reverse)


def _expand_grid(search_space: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not search_space:
        return []
    keys = sorted(search_space)
    values = []
    for key in keys:
        raw = search_space[key]
        values.append(raw if isinstance(raw, list) else [raw])
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _trial_specs(stage_cfg: Mapping[str, Any], prefix: str) -> List[Dict[str, Any]]:
    explicit = list(stage_cfg.get("trials", []) or stage_cfg.get("strategies", []) or [])
    specs: List[Dict[str, Any]] = []
    for idx, item in enumerate(explicit):
        name = str(item.get("name") or f"{prefix}_{idx:03d}")
        overrides = {key: value for key, value in item.items() if key != "name"}
        if "training" in item:
            overrides = dict(item["training"])
        if "strategy" in item:
            overrides = dict(item["strategy"])
        specs.append({"id": _slug(name), "overrides": overrides})

    search_key = "search_space" if "search_space" in stage_cfg else "strategy_search"
    for idx, overrides in enumerate(_expand_grid(stage_cfg.get(search_key, {}) or {}), start=len(specs)):
        specs.append({"id": f"{prefix}_{idx:03d}", "overrides": overrides})

    if not specs:
        specs.append({"id": f"{prefix}_000", "overrides": {}})

    max_trials = stage_cfg.get("max_trials")
    if max_trials is not None:
        specs = specs[: int(max_trials)]
    return specs


def _raw_data_manifest(config: Mapping[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    granularity = data_cfg.get("granularity", DEFAULT_GRANULARITY)
    raw_dir = Path(data_cfg.get("dir", "data")) / "raw"
    files = sorted(raw_dir.glob(f"ETHUSDT-{granularity}-lighter-*.csv"))
    return {
        "raw_dir": raw_dir,
        "granularity": granularity,
        "files": [
            {
                "path": path,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }


def _split_indices(n_samples: int, split_cfg: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    if n_samples < 3:
        raise ValueError(f"Need at least 3 samples for staged train/validation/test split; got {n_samples}")
    train_fraction = float(split_cfg.get("train_fraction", 0.6))
    validation_fraction = float(split_cfg.get("validation_fraction", 0.2))
    purge = int(split_cfg.get("purge_bars", 0) or 0)
    embargo = int(split_cfg.get("embargo_bars", 0) or 0)

    train_end = max(1, min(n_samples - 2, int(n_samples * train_fraction)))
    val_end = max(train_end + 1, min(n_samples - 1, train_end + int(n_samples * validation_fraction)))

    train_stop = max(1, train_end - purge)
    val_start = min(val_end - 1, train_end + embargo)
    test_start = min(n_samples - 1, val_end + embargo)

    splits = {
        "train": np.arange(0, train_stop, dtype=int),
        "validation": np.arange(val_start, val_end, dtype=int),
        "test": np.arange(test_start, n_samples, dtype=int),
    }
    if any(len(indices) == 0 for indices in splits.values()):
        splits = {
            "train": np.arange(0, train_end, dtype=int),
            "validation": np.arange(train_end, val_end, dtype=int),
            "test": np.arange(val_end, n_samples, dtype=int),
        }
    if any(len(indices) == 0 for indices in splits.values()):
        raise ValueError(f"Unable to build non-empty staged splits from {n_samples} samples")
    return splits


def _dataset_subset(dataset: Mapping[str, Any], indices: np.ndarray) -> Dict[str, Any]:
    return {
        "X": dataset["X"][indices],
        "y_ret": dataset["y_ret"][indices],
        "y_dir": dataset["y_dir"][indices],
        "sample_weights": dataset["sample_weights"][indices],
    }


def _predict_batches(model: torch.nn.Module, X: torch.Tensor, device: torch.device, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    predictions: List[torch.Tensor] = []
    confidences: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size].to(device)
            pred, conf = model(batch, return_confidence=True)
            predictions.append(pred.detach().cpu())
            confidences.append(conf.detach().cpu())
    return torch.cat(predictions, dim=0), torch.cat(confidences, dim=0)


def _prediction_frame(dataset: Mapping[str, Any], indices: np.ndarray, prediction: torch.Tensor, confidence: torch.Tensor) -> pd.DataFrame:
    closes = dataset["close"].iloc[indices]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(closes.index),
            "close": closes.to_numpy(dtype=float),
            "y_ret": dataset["y_ret"][indices].detach().cpu().numpy().reshape(-1),
            "y_dir": dataset["y_dir"][indices].detach().cpu().numpy().reshape(-1),
            "prediction": prediction.detach().cpu().numpy().reshape(-1),
            "confidence": confidence.detach().cpu().numpy().reshape(-1),
        }
    )


def _evaluate_predictions(y_ret: torch.Tensor, y_dir: torch.Tensor, prediction: torch.Tensor, confidence: torch.Tensor) -> Dict[str, Dict[str, float]]:
    confidence_targets = (y_dir != 0).float().unsqueeze(1)
    return {
        "prediction": _sanitize_metrics(compute_metrics(y_ret, prediction, "regression")),
        "confidence": _sanitize_metrics(compute_metrics(confidence_targets, confidence, "classification")),
    }


def _write_stage0(config: Dict[str, Any], staged: Dict[str, Any], dataset: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    stage_dir = run_dir / "stage0_features"
    stage_dir.mkdir(parents=True, exist_ok=True)
    granularity = dataset["granularity"]
    preprocessor = DataPreprocessor(data_dir=config["data"]["dir"], granularities=[granularity])
    close = dataset["close"]
    timestamps_path = stage_dir / "timestamps.parquet"
    pd.DataFrame({"timestamp": pd.to_datetime(close.index), "close": close.to_numpy(dtype=float)}).to_parquet(timestamps_path, index=False)

    tensor_paths: Dict[str, str] = {}
    if staged.get("save_stage0_tensors"):
        for key in ["X", "y_ret", "y_dir", "sample_weights"]:
            path = stage_dir / f"{key}.pt"
            torch.save(dataset[key], path)
            tensor_paths[key] = str(path)

    manifest = {
        "stage": "stage0_features",
        "path": stage_dir,
        "target": dataset["target"],
        "granularity": granularity,
        "sequence_length": dataset["sequence_length"],
        "timeout_bars": dataset["timeout_bars"],
        "next_hour_horizon_bars": dataset["next_hour_horizon_bars"],
        "feature_columns": preprocessor.get_feature_cols(),
        "feature_window_bars": preprocessor.feature_window_bars(granularity),
        "input_size": dataset["input_size"],
        "samples": int(dataset["X"].shape[0]),
        "X_shape": list(dataset["X"].shape),
        "y_ret_shape": list(dataset["y_ret"].shape),
        "close_coverage": {"start": close.index[0], "end": close.index[-1]},
        "timestamps_path": timestamps_path,
        "tensor_paths": tensor_paths,
        "raw_data": _raw_data_manifest(config),
    }
    _write_json(stage_dir / "manifest.json", manifest)
    return manifest


def _write_split_manifest(dataset: Mapping[str, Any], splits: Mapping[str, np.ndarray], path: Path) -> Dict[str, Any]:
    close = dataset["close"]
    manifest = {"splits": {}}
    for name, indices in splits.items():
        series = close.iloc[indices]
        manifest["splits"][name] = {
            "samples": int(len(indices)),
            "start_index": int(indices[0]),
            "end_index": int(indices[-1]),
            "start_timestamp": series.index[0],
            "end_timestamp": series.index[-1],
        }
    _write_json(path, manifest)
    return manifest


def _run_stage1(
    config: Dict[str, Any],
    staged: Dict[str, Any],
    dataset: Dict[str, Any],
    splits: Dict[str, np.ndarray],
    run_dir: Path,
    *,
    smoke: bool,
    device: torch.device,
    allow_cpu: bool,
) -> Dict[str, Any]:
    stage_dir = run_dir / "stage1_model"
    trials_dir = stage_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    base_training = _training_config(config, smoke)
    specs = _trial_specs(staged.get("stage1", {}), "model_trial")
    batch_size = int(staged.get("stage1", {}).get("prediction_batch_size", 512))
    trial_manifests: List[Dict[str, Any]] = []

    for idx, spec in enumerate(specs):
        trial_id = spec["id"]
        trial_dir = trials_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        training_cfg = _deep_update(dict(base_training), spec["overrides"])
        torch.manual_seed(int(config.get("glft", {}).get("seed", 42)) + idx)
        np.random.seed(int(config.get("glft", {}).get("seed", 42)) + idx)

        train_data = _dataset_subset(dataset, splits["train"])
        results = hierarchical_training_pipeline(
            train_data["X"],
            train_data["y_ret"],
            train_data["y_dir"],
            train_data["sample_weights"],
            input_size=int(dataset["input_size"]),
            hidden_size=int(training_cfg["hidden_size"]),
            num_layers=int(training_cfg["num_layers"]),
            dropout=float(training_cfg["dropout"]),
            batch_size=int(training_cfg["batch_size"]),
            num_epochs=int(training_cfg["epochs"]),
            learning_rate=float(training_cfg["learning_rate"]),
            device=device,
            allow_cpu=allow_cpu,
        )

        model = results["hierarchical"]["model"]
        model_paths = {
            "hierarchical": trial_dir / "hierarchical_model.pt",
            "level_0": trial_dir / "model_level_0.pt",
            "level_1": trial_dir / "model_level_1.pt",
            "level_2": trial_dir / "model_level_2.pt",
        }
        torch.save(model.state_dict(), model_paths["hierarchical"])
        torch.save(results["level_0"]["model"].state_dict(), model_paths["level_0"])
        torch.save(results["level_1"]["model"].state_dict(), model_paths["level_1"])
        torch.save(results["level_2"]["model"].state_dict(), model_paths["level_2"])

        metrics: Dict[str, Any] = {}
        prediction_paths: Dict[str, str] = {}
        for split_name, indices in splits.items():
            subset = _dataset_subset(dataset, indices)
            pred, conf = _predict_batches(model, subset["X"], device, batch_size=batch_size)
            metrics[split_name] = _evaluate_predictions(subset["y_ret"], subset["y_dir"], pred, conf)
            frame = _prediction_frame(dataset, indices, pred, conf)
            pred_path = trial_dir / f"predictions_{split_name}.parquet"
            frame.to_parquet(pred_path, index=False)
            prediction_paths[split_name] = str(pred_path)

        training_history = {
            key: {
                "final_loss": _safe_float(value.get("final_loss")),
                "training_losses": [_safe_float(item) for item in value.get("training_losses", [])],
            }
            for key, value in results.items()
            if key.startswith("level_")
        }
        _write_json(trial_dir / "training_history.json", training_history)

        manifest = {
            "stage": "stage1_model",
            "trial_index": idx,
            "trial_id": trial_id,
            "target": dataset["target"],
            "training": training_cfg,
            "architecture": {
                "input_size": int(dataset["input_size"]),
                "hidden_size": int(training_cfg["hidden_size"]),
                "num_layers": int(training_cfg["num_layers"]),
                "dropout": float(training_cfg["dropout"]),
                "sequence_length": int(dataset["sequence_length"]),
                "granularity": dataset["granularity"],
            },
            "model_paths": model_paths,
            "prediction_paths": prediction_paths,
            "metrics": metrics,
            "training_history_path": trial_dir / "training_history.json",
            "manifest_path": trial_dir / "manifest.json",
        }
        _write_json(trial_dir / "manifest.json", manifest)
        trial_manifests.append(manifest)

    ranked = _rank_payloads(trial_manifests, staged["selection_metric"], staged["selection_mode"])
    best = dict(ranked[0])
    best["selected_by"] = {
        "metric": staged["selection_metric"],
        "mode": staged["selection_mode"],
        "value": _metric_value(best, staged["selection_metric"]),
    }
    best_path = stage_dir / "best_model_manifest.json"
    _write_json(best_path, best)
    summary = {
        "stage": "stage1_model",
        "path": stage_dir,
        "trials": trial_manifests,
        "best_model_manifest_path": best_path,
        "best_trial_id": best["trial_id"],
        "selection": best["selected_by"],
    }
    _write_json(stage_dir / "manifest.json", summary)
    return summary


def _load_best_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_specs(config: Dict[str, Any], staged: Dict[str, Any]) -> List[Dict[str, Any]]:
    glft = config.get("glft", {})
    stage2 = staged.get("stage2", {})
    defaults = {
        "initial_capital": glft.get("initial_capital", 100_000.0),
        "gamma": glft.get("gamma", 0.5),
        "kappa": glft.get("kappa", 0.1),
        "dt": glft.get("dt", 1.0 / bars_for_duration(config.get("data", {}).get("granularity", DEFAULT_GRANULARITY), hours=24)),
        "max_inventory": glft.get("max_inventory", 10.0),
        "min_spread": glft.get("min_spread", 5.0),
        "max_drawdown": glft.get("max_drawdown", 0.2),
        "seed": glft.get("seed", 42),
    }
    defaults = _deep_update(defaults, stage2.get("strategy", {}) or {})
    stage_cfg = {
        "trials": stage2.get("strategies", []) or [],
        "search_space": stage2.get("strategy_search", {}) or {},
        "max_trials": stage2.get("max_trials"),
    }
    specs = _trial_specs(stage_cfg, "strategy_trial")
    return [{"id": spec["id"], "strategy": _deep_update(dict(defaults), spec["overrides"])} for spec in specs]


def _backtest_one_strategy(config: Dict[str, Any], staged: Dict[str, Any], pred_df: pd.DataFrame, strategy: Dict[str, Any]) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_df = pred_df.copy()
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    max_rows = staged.get("stage2", {}).get("max_rows")
    if max_rows:
        pred_df = pred_df.tail(int(max_rows))
    pred_df = pred_df.set_index("timestamp")

    clip = float(staged.get("stage2", {}).get("prediction_clip", 0.05))
    predicted_returns = np.clip(pred_df["prediction"].astype(float), -clip, clip)
    predicted_prices = pred_df["close"].astype(float) * np.exp(predicted_returns)
    market_data = pd.DataFrame({"price": pred_df["close"].astype(float)}, index=pred_df.index)

    granularity = config.get("data", {}).get("granularity", DEFAULT_GRANULARITY)
    one_hour_bars = bars_for_duration(granularity, hours=1)
    volatility = market_data["price"].pct_change().rolling(one_hour_bars, min_periods=2).std()
    fallback_vol = _safe_float(volatility.mean(), 0.001)
    volatility = volatility.fillna(fallback_vol)

    params = GLFTParams(
        gamma=float(strategy["gamma"]),
        kappa=float(strategy["kappa"]),
        sigma=max(fallback_vol, 1e-6),
        dt=float(strategy["dt"]),
        max_inventory=float(strategy["max_inventory"]),
        min_spread=float(strategy["min_spread"]),
    )
    backtester = GLFTBacktester(
        BacktestParams(
            start_date=pd.to_datetime(market_data.index[0]).to_pydatetime(),
            end_date=pd.to_datetime(market_data.index[-1]).to_pydatetime(),
            initial_capital=float(strategy["initial_capital"]),
            seed=int(strategy["seed"]),
            gamma=float(strategy["gamma"]),
            inventory_limit=float(strategy["max_inventory"]),
            quote_spread=float(strategy["min_spread"]),
        ),
        GLFTQuoteCalculator(params),
        InventoryBook(max_position=float(strategy["max_inventory"]), max_drawdown=float(strategy["max_drawdown"])),
    )
    result = backtester.run(market_data, predicted_prices, volatility=volatility)
    metrics = _sanitize_metrics(result.metrics)
    if "num_trades" not in metrics:
        metrics.update({"num_trades": 0.0, "net_pnl": 0.0, "total_pnl": 0.0, "total_fees": 0.0})

    trades = pd.DataFrame(result.trades)
    if not trades.empty:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        trades["net_pnl"] = trades["pnl"] - trades["fee"]
        trades["cumulative_net_pnl"] = trades["net_pnl"].cumsum()
    else:
        trades = pd.DataFrame(columns=["timestamp", "side", "price", "size", "fee", "pnl", "net_pnl", "cumulative_net_pnl"])

    series = pd.DataFrame(
        {
            "timestamp": market_data.index,
            "close": market_data["price"].to_numpy(dtype=float),
            "predicted_return": predicted_returns.to_numpy(dtype=float),
            "predicted_price": predicted_prices.to_numpy(dtype=float),
            "volatility": volatility.to_numpy(dtype=float),
            "inventory": result.inventory_history[: len(market_data)],
            "spread": result.spread_history[: len(market_data)],
        }
    )
    pnl = trades[["timestamp", "net_pnl", "cumulative_net_pnl"]].copy()
    return metrics, trades, series, pnl


def _run_stage2(config: Dict[str, Any], staged: Dict[str, Any], stage1: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    stage_dir = run_dir / "stage2_backtest"
    strategies_dir = stage_dir / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    best_manifest_path = Path(stage1["best_model_manifest_path"])
    best = _load_best_manifest(best_manifest_path)
    pred_path = Path(best["prediction_paths"]["test"])
    pred_df = pd.read_parquet(pred_path)

    strategy_manifests: List[Dict[str, Any]] = []
    for idx, spec in enumerate(_strategy_specs(config, staged)):
        strategy_id = spec["id"]
        strategy_dir = strategies_dir / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)
        metrics, trades, series, pnl = _backtest_one_strategy(config, staged, pred_df, spec["strategy"])
        trades_path = strategy_dir / "trades.parquet"
        series_path = strategy_dir / "timeseries.parquet"
        pnl_path = strategy_dir / "pnl.parquet"
        trades.to_parquet(trades_path, index=False)
        series.to_parquet(series_path, index=False)
        pnl.to_parquet(pnl_path, index=False)
        manifest = {
            "stage": "stage2_backtest",
            "strategy_index": idx,
            "strategy_id": strategy_id,
            "strategy": spec["strategy"],
            "consumed_model_manifest": best_manifest_path,
            "consumed_predictions": pred_path,
            "metrics": metrics,
            "trades_path": trades_path,
            "timeseries_path": series_path,
            "pnl_path": pnl_path,
            "manifest_path": strategy_dir / "manifest.json",
        }
        _write_json(strategy_dir / "manifest.json", manifest)
        strategy_manifests.append(manifest)

    stage2_cfg = staged.get("stage2", {})
    ranked = _rank_payloads(strategy_manifests, stage2_cfg.get("selection_metric", "metrics.net_pnl"), stage2_cfg.get("selection_mode", "max"))
    best_strategy = dict(ranked[0])
    best_strategy["selected_by"] = {
        "metric": stage2_cfg.get("selection_metric", "metrics.net_pnl"),
        "mode": stage2_cfg.get("selection_mode", "max"),
        "value": _metric_value(best_strategy, stage2_cfg.get("selection_metric", "metrics.net_pnl")),
    }
    _write_json(stage_dir / "best_strategy_manifest.json", best_strategy)
    summary = {
        "stage": "stage2_backtest",
        "path": stage_dir,
        "consumed_model_manifest": best_manifest_path,
        "consumed_predictions": pred_path,
        "strategies": strategy_manifests,
        "best_strategy_manifest_path": stage_dir / "best_strategy_manifest.json",
        "best_strategy_id": best_strategy["strategy_id"],
        "selection": best_strategy["selected_by"],
    }
    _write_json(stage_dir / "manifest.json", summary)
    return summary


def _report_markdown(report: Mapping[str, Any]) -> str:
    stage1 = report["stage1"]
    stage2 = report["stage2"]
    best_model_path = Path(stage1["best_model_manifest_path"])
    best_strategy_path = Path(stage2["best_strategy_manifest_path"])
    best_model = _load_best_manifest(best_model_path)
    best_strategy = _load_best_manifest(best_strategy_path)
    lines = [
        f"# Staged Trial Report: {report['run_id']}",
        "",
        f"- Target: `{report['target']}`",
        f"- Granularity: `{report['stage0']['granularity']}`",
        f"- Samples: `{report['stage0']['samples']}`",
        f"- Sequence length: `{report['stage0']['sequence_length']}` bars",
        f"- Stage 1 trials: `{len(stage1['trials'])}`",
        f"- Best model trial: `{stage1['best_trial_id']}` selected by `{stage1['selection']['metric']}` = `{stage1['selection']['value']}`",
        f"- Stage 2 strategies: `{len(stage2['strategies'])}`",
        f"- Best strategy: `{stage2['best_strategy_id']}` selected by `{stage2['selection']['metric']}` = `{stage2['selection']['value']}`",
        "",
        "## Best Model Validation Metrics",
        "",
    ]
    for group, metrics in best_model["metrics"]["validation"].items():
        lines.append(f"### {group}")
        for key, value in metrics.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(["## Best Strategy Backtest Metrics", ""])
    for key, value in best_strategy["metrics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Key Artifacts",
            "",
            f"- Stage 0 manifest: `{report['stage0_manifest_path']}`",
            f"- Best model manifest: `{stage1['best_model_manifest_path']}`",
            f"- Best strategy manifest: `{stage2['best_strategy_manifest_path']}`",
            f"- Report JSON: `{report['report_json_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_report(run_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / "report.json"
    report_md = report_dir / "report.md"
    payload = dict(payload)
    payload["report_json_path"] = report_json
    payload["report_markdown_path"] = report_md
    _write_json(report_json, payload)
    report_md.write_text(_report_markdown(payload), encoding="utf-8")
    return {"report_json_path": report_json, "report_markdown_path": report_md}


def run_staged_trial(
    config_path: Path,
    *,
    smoke: bool = False,
    device: str = "cuda:0",
    allow_cpu: bool = False,
    run_name: Optional[str] = None,
    artifact_root: Optional[Path] = None,
) -> Dict[str, Any]:
    config = load_experiment_config(config_path)
    targets_cfg = config.get("targets")
    if config.get("version") == 2 or (isinstance(targets_cfg, Mapping) and "horizons" in targets_cfg):
        from src.experiments.meta_labeling_mvp import run_meta_labeling_mvp

        return run_meta_labeling_mvp(
            config_path,
            smoke=smoke,
            device=device,
            allow_cpu=allow_cpu,
            run_name=run_name,
            artifact_root=artifact_root,
        )

    staged = _staged_config(config, smoke=smoke)
    if run_name:
        staged["run_name"] = run_name
    if artifact_root:
        staged["artifact_root"] = str(artifact_root)

    run_id = _run_id(str(staged["run_name"]))
    run_dir = Path(staged["artifact_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    target = str(staged.get("target") or config.get("targets", ["next_hour_return"])[0])
    resolved_device = resolve_training_device(device, allow_cpu=allow_cpu)

    resolved_config = deepcopy(config)
    resolved_config["staged_trial"] = staged
    _write_yaml(run_dir / "resolved_config.yml", resolved_config)
    _write_json(run_dir / "environment.json", _environment_manifest())

    dataset = build_lighter_dataset(config, target, smoke=smoke)
    stage0 = _write_stage0(config, staged, dataset, run_dir)
    stage0_manifest_path = run_dir / "stage0_features" / "manifest.json"
    splits = _split_indices(int(dataset["X"].shape[0]), staged.get("split", {}))
    split_manifest = _write_split_manifest(dataset, splits, run_dir / "stage1_model" / "split_manifest.json")
    stage1 = _run_stage1(
        config,
        staged,
        dataset,
        splits,
        run_dir,
        smoke=smoke,
        device=resolved_device,
        allow_cpu=allow_cpu,
    )
    stage2 = _run_stage2(config, staged, stage1, run_dir)

    report = {
        "run_id": run_id,
        "run_dir": run_dir,
        "config_path": config_path,
        "smoke": smoke,
        "device": str(resolved_device),
        "target": target,
        "stage0": stage0,
        "stage0_manifest_path": stage0_manifest_path,
        "split_manifest": split_manifest,
        "stage1": stage1,
        "stage2": stage2,
        "status": "success",
    }
    report_paths = _write_report(run_dir, report)
    result = {**report, **report_paths}
    _write_json(run_dir / "manifest.json", result)
    return _json_ready(result)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a staged Lighter model/backtest trial with artifact tracking.")
    parser.add_argument("--config", type=Path, default=Path("configs/staged_trial_smoke.yml"))
    parser.add_argument("--smoke", action="store_true", help="Apply smoke-sized config overrides.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run_staged_trial(
        args.config,
        smoke=args.smoke,
        device=args.device,
        allow_cpu=args.allow_cpu,
        run_name=args.run_name,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
