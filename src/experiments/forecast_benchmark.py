"""Forecast benchmark layer for the v2 ETHPredict research pipeline.

The benchmark intentionally separates forecast quality from trading PnL. Each
model emits the same multi-horizon prediction schema, then the runner optionally
routes those predictions through the existing meta-label/backtest layer.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from src.experiments.meta_labeling_mvp import (
    _json_ready,
    _load_yaml,
    _prediction_metrics,
    _run_id,
    _total_cost_bps,
    _write_json,
    _write_yaml,
    add_meta_labels,
    add_meta_probabilities,
    apply_smoke_overrides,
    build_multi_horizon_lighter_dataset,
    candidate_signals,
    fit_meta_labeler,
    predict_base_model,
    purged_walk_forward_splits,
    run_alpha_backtest,
    target_horizons,
    train_base_model,
)
from src.training.devices import resolve_training_device
from src.utils.trackio_logging import log_trackio_run


DEFAULT_BENCHMARK_MODELS = ["zero_return", "momentum", "lstm"]
DEFAULT_TIMESFM_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"


class TimesFMUnavailable(RuntimeError):
    """Raised when TimesFM cannot be imported, loaded, or executed."""


def _benchmark_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = deepcopy(dict(config.get("benchmark", {}) or {}))
    raw.setdefault("enabled", True)
    raw.setdefault("models", list(DEFAULT_BENCHMARK_MODELS))
    raw.setdefault("route_to_meta_backtest", True)
    raw.setdefault("prediction_batch_size", int(config.get("training", {}).get("prediction_batch_size", 512)))
    raw.setdefault("momentum", {})
    raw.setdefault("lstm", {})
    raw.setdefault("timesfm", {})
    return raw


def _standard_frame(dataset: Mapping[str, Any], indices: np.ndarray) -> pd.DataFrame:
    frame = dataset["samples"].iloc[indices].reset_index(drop=True).copy()
    true_ret = dataset["y_ret"][indices].detach().cpu().numpy()
    true_dir = dataset["y_dir"][indices].detach().cpu().numpy()
    for h_idx, horizon in enumerate(dataset["horizon_names"]):
        frame[f"{horizon}_true_return"] = true_ret[:, h_idx]
        frame[f"{horizon}_true_direction"] = true_dir[:, h_idx]
    return frame


def _direction_probability(pred_return: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.asarray(scale, dtype=float), 1e-8)
    logits = np.clip(np.asarray(pred_return, dtype=float) / scale, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-logits))


def zero_return_predictions(dataset: Mapping[str, Any], indices: np.ndarray) -> pd.DataFrame:
    """Naive last-value baseline: predicted price is unchanged, so return is 0."""

    frame = _standard_frame(dataset, indices)
    for horizon in dataset["horizon_names"]:
        frame[f"{horizon}_pred_return"] = 0.0
        frame[f"{horizon}_direction_prob"] = 0.5
    return frame


def momentum_predictions(dataset: Mapping[str, Any], indices: np.ndarray) -> pd.DataFrame:
    """Simple momentum baseline using the previous horizon-length log return."""

    frame = _standard_frame(dataset, indices)
    close = dataset["price_path"]["close"].astype(float).reset_index(drop=True)
    sample_index = frame["sample_index"].astype(int).to_numpy()
    realized_vol = frame.get("realized_vol", pd.Series(0.001, index=frame.index)).to_numpy(dtype=float)
    for horizon, bars in dataset["horizon_bars"].items():
        pred = np.zeros(len(frame), dtype=float)
        for row_idx, label_idx in enumerate(sample_index):
            prior_idx = max(0, int(label_idx) - int(bars))
            current = float(close.iloc[int(label_idx)])
            prior = float(close.iloc[prior_idx])
            pred[row_idx] = math.log(current / prior) if current > 0 and prior > 0 and prior_idx != label_idx else 0.0
        frame[f"{horizon}_pred_return"] = pred
        frame[f"{horizon}_direction_prob"] = _direction_probability(pred, realized_vol)
    return frame


class TimesFMZeroShotAdapter:
    """Small compatibility wrapper around TimesFM zero-shot forecasting.

    The current TimesFM package exposes the v2.5 PyTorch model through
    ``TimesFM_2p5_200M_torch``. Older package versions exposed ``TimesFm`` with
    hparams/checkpoint objects. This adapter supports both shapes and allows a
    fake model to be injected for unit tests.
    """

    def __init__(
        self,
        *,
        context_length: int = 1024,
        horizon_length: int = 12,
        checkpoint: str = DEFAULT_TIMESFM_CHECKPOINT,
        input_series: str = "log_close",
        batch_size: int = 16,
        model: Optional[Any] = None,
        compile_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        if input_series not in {"close", "log_close"}:
            raise ValueError("TimesFM input_series must be 'close' or 'log_close'")
        self.context_length = int(context_length)
        self.horizon_length = int(horizon_length)
        self.checkpoint = str(checkpoint)
        self.input_series = input_series
        self.batch_size = int(batch_size)
        self._model = model
        self.compile_kwargs = dict(compile_kwargs or {})

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        try:
            import timesfm  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise TimesFMUnavailable(f"TimesFM package is not installed: {exc}") from exc

        if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            try:
                model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
                config_cls = getattr(timesfm, "ForecastConfig")
                compile_kwargs = {
                    "max_context": self.context_length,
                    "max_horizon": self.horizon_length,
                    "normalize_inputs": True,
                    "use_continuous_quantile_head": True,
                    "force_flip_invariance": True,
                    "infer_is_positive": self.input_series == "close",
                    "fix_quantile_crossing": True,
                }
                compile_kwargs.update(self.compile_kwargs)
                model.compile(config_cls(**compile_kwargs))
                return model
            except Exception as exc:  # pragma: no cover - optional package path
                raise TimesFMUnavailable(f"Failed to load TimesFM v2.5 checkpoint {self.checkpoint!r}: {exc}") from exc

        if all(hasattr(timesfm, name) for name in ["TimesFm", "TimesFmHparams", "TimesFmCheckpoint"]):
            try:
                hparams = timesfm.TimesFmHparams(
                    backend="cpu",
                    context_len=self.context_length,
                    horizon_len=self.horizon_length,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=50,
                    model_dims=1280,
                )
                checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.checkpoint)
                return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
            except Exception as exc:  # pragma: no cover - optional package path
                raise TimesFMUnavailable(f"Failed to load legacy TimesFM checkpoint {self.checkpoint!r}: {exc}") from exc

        raise TimesFMUnavailable("Installed timesfm package does not expose a supported forecasting API")

    def _forecast_batch(self, inputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        model = self.model
        horizon = self.horizon_length
        try:
            forecast = model.forecast(horizon=horizon, inputs=list(inputs))
        except TypeError:
            try:
                forecast = model.forecast(list(inputs), horizon=horizon)
            except TypeError:
                forecast = model.forecast(list(inputs), freq=[0] * len(inputs))
        if isinstance(forecast, tuple):
            point = np.asarray(forecast[0], dtype=float)
            quantiles = np.asarray(forecast[1], dtype=float) if len(forecast) > 1 and forecast[1] is not None else None
        else:
            point = np.asarray(forecast, dtype=float)
            quantiles = None
        if point.ndim != 2:
            raise TimesFMUnavailable(f"TimesFM point forecast must be 2D, got shape {point.shape}")
        return point, quantiles

    def forecast_returns(self, close: pd.Series, sample_indices: Sequence[int], horizon_bars: Mapping[str, int]) -> Dict[str, Any]:
        close = close.astype(float).reset_index(drop=True)
        sample_indices = [int(idx) for idx in sample_indices]
        max_horizon = max(int(bars) for bars in horizon_bars.values())
        if max_horizon > self.horizon_length:
            self.horizon_length = max_horizon

        contexts: List[np.ndarray] = []
        anchors: List[float] = []
        for idx in sample_indices:
            start = max(0, idx - self.context_length + 1)
            raw = close.iloc[start : idx + 1].to_numpy(dtype=float)
            raw = raw[np.isfinite(raw) & (raw > 0)]
            if len(raw) == 0:
                raw = np.array([1.0], dtype=float)
            anchors.append(float(raw[-1]))
            context = np.log(raw) if self.input_series == "log_close" else raw
            contexts.append(context.astype(float))

        point_batches: List[np.ndarray] = []
        quantile_batches: List[np.ndarray] = []
        has_quantiles = False
        for start in range(0, len(contexts), self.batch_size):
            point, quantiles = self._forecast_batch(contexts[start : start + self.batch_size])
            point_batches.append(point)
            if quantiles is not None:
                has_quantiles = True
                quantile_batches.append(quantiles)

        point_all = np.concatenate(point_batches, axis=0) if point_batches else np.empty((0, max_horizon))
        quantile_all = np.concatenate(quantile_batches, axis=0) if has_quantiles and quantile_batches else None
        anchors_arr = np.asarray(anchors, dtype=float)

        returns: Dict[str, np.ndarray] = {}
        quantile_returns: Dict[str, Dict[str, np.ndarray]] = {}
        for horizon, bars in horizon_bars.items():
            col_idx = int(bars) - 1
            if self.input_series == "log_close":
                returns[horizon] = point_all[:, col_idx] - np.log(anchors_arr)
            else:
                forecast_price = np.maximum(point_all[:, col_idx], 1e-12)
                returns[horizon] = np.log(forecast_price / np.maximum(anchors_arr, 1e-12))

            if quantile_all is not None and quantile_all.ndim == 3 and quantile_all.shape[1] > col_idx:
                quantile_returns[horizon] = {}
                for label, q_idx in _selected_quantile_indices(quantile_all.shape[2]).items():
                    raw_q = quantile_all[:, col_idx, q_idx]
                    if self.input_series == "log_close":
                        quantile_returns[horizon][label] = raw_q - np.log(anchors_arr)
                    else:
                        quantile_returns[horizon][label] = np.log(np.maximum(raw_q, 1e-12) / np.maximum(anchors_arr, 1e-12))

        return {"returns": returns, "quantile_returns": quantile_returns}


def _selected_quantile_indices(width: int) -> Dict[str, int]:
    if width >= 10:
        return {"q10": 1, "q50": 5, "q90": 9}
    if width >= 9:
        return {"q10": 0, "q50": 4, "q90": 8}
    if width >= 3:
        return {"q10": 0, "q50": width // 2, "q90": width - 1}
    return {}


def timesfm_predictions(dataset: Mapping[str, Any], indices: np.ndarray, adapter: TimesFMZeroShotAdapter) -> pd.DataFrame:
    frame = _standard_frame(dataset, indices)
    forecasts = adapter.forecast_returns(
        dataset["price_path"]["close"],
        frame["sample_index"].astype(int).to_list(),
        dataset["horizon_bars"],
    )
    realized_vol = frame.get("realized_vol", pd.Series(0.001, index=frame.index)).to_numpy(dtype=float)
    for horizon in dataset["horizon_names"]:
        pred = np.asarray(forecasts["returns"][horizon], dtype=float)
        frame[f"{horizon}_pred_return"] = pred
        frame[f"{horizon}_direction_prob"] = _direction_probability(pred, realized_vol)
        for label, values in forecasts.get("quantile_returns", {}).get(horizon, {}).items():
            frame[f"{horizon}_pred_return_{label}"] = np.asarray(values, dtype=float)
    return frame


def _edge_bucket_table(
    true_return: np.ndarray,
    pred_return: np.ndarray,
    *,
    cost_bps: float,
    bins: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    bins = list(bins or [-math.inf, 0.0, 5.0, 10.0, 25.0, 50.0, math.inf])
    labels = ["lt_0", "0_5", "5_10", "10_25", "25_50", "50_plus"]
    pred_edge = np.abs(pred_return) * 10_000.0 - cost_bps
    side = np.sign(pred_return)
    realized_net_edge = side * true_return * 10_000.0 - cost_bps
    correct = np.sign(true_return) == side
    buckets = pd.cut(pred_edge, bins=bins, labels=labels, include_lowest=True)
    table = pd.DataFrame(
        {
            "bucket": buckets,
            "predicted_edge_bps": pred_edge,
            "realized_net_edge_bps": realized_net_edge,
            "correct_direction": correct,
        }
    )
    rows: List[Dict[str, Any]] = []
    for label in labels:
        part = table[table["bucket"] == label]
        rows.append(
            {
                "bucket": label,
                "count": int(len(part)),
                "mean_predicted_edge_bps": float(part["predicted_edge_bps"].mean()) if len(part) else 0.0,
                "mean_realized_net_edge_bps": float(part["realized_net_edge_bps"].mean()) if len(part) else 0.0,
                "directional_accuracy": float(part["correct_direction"].mean()) if len(part) else 0.0,
            }
        )
    return rows


def forecast_metrics(frame: pd.DataFrame, dataset: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = _prediction_metrics(frame, dataset["horizon_names"])
    for horizon in dataset["horizon_names"]:
        true_return = frame[f"{horizon}_true_return"].to_numpy(dtype=float)
        pred_return = frame[f"{horizon}_pred_return"].to_numpy(dtype=float)
        cost_bps = _total_cost_bps(config, int(dataset["horizon_bars"][horizon]), str(dataset["granularity"]))
        side = np.sign(pred_return)
        realized_net_edge = side * true_return * 10_000.0 - cost_bps
        metrics[horizon]["edge"] = {
            "cost_bps": float(cost_bps),
            "mean_predicted_edge_bps": float(np.mean(np.abs(pred_return) * 10_000.0 - cost_bps)) if len(pred_return) else 0.0,
            "mean_realized_net_edge_bps": float(np.mean(realized_net_edge)) if len(realized_net_edge) else 0.0,
            "positive_realized_net_edge_rate": float(np.mean(realized_net_edge > 0.0)) if len(realized_net_edge) else 0.0,
            "predicted_edge_buckets": _edge_bucket_table(true_return, pred_return, cost_bps=cost_bps),
        }
    return metrics


def _run_meta_backtest_for_predictions(
    model_id: str,
    oof_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    model_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    oof_candidates = add_meta_labels(candidate_signals(oof_predictions, dataset, config), dataset, config)
    test_candidates = add_meta_labels(candidate_signals(test_predictions, dataset, config), dataset, config)
    meta_model = fit_meta_labeler(oof_candidates, config, dataset["horizon_names"], device)
    oof_candidates = add_meta_probabilities(oof_candidates, meta_model)
    test_candidates = add_meta_probabilities(test_candidates, meta_model)
    validation_metrics, validation_trades = run_alpha_backtest(oof_candidates, config)
    test_metrics, test_trades = run_alpha_backtest(test_candidates, config)

    paths = {
        "oof_candidates": model_dir / "meta_candidates_oof.parquet",
        "test_candidates": model_dir / "meta_candidates_test.parquet",
        "validation_trades": model_dir / "alpha_trades_validation.parquet",
        "test_trades": model_dir / "alpha_trades_test.parquet",
    }
    oof_candidates.to_parquet(paths["oof_candidates"], index=False)
    test_candidates.to_parquet(paths["test_candidates"], index=False)
    validation_trades.to_parquet(paths["validation_trades"], index=False)
    test_trades.to_parquet(paths["test_trades"], index=False)

    return {
        "meta_labeler": meta_model.manifest(),
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "artifact_paths": paths,
    }


def _write_model_result(
    model_id: str,
    model_dir: Path,
    oof_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    kind: str,
    status: str = "success",
    reason: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "oof_predictions": model_dir / "predictions_oof.parquet",
        "test_predictions": model_dir / "predictions_test.parquet",
    }
    oof_predictions.to_parquet(paths["oof_predictions"], index=False)
    test_predictions.to_parquet(paths["test_predictions"], index=False)

    metrics = {
        "forecast": {
            "validation": forecast_metrics(oof_predictions, dataset, config),
            "test": forecast_metrics(test_predictions, dataset, config),
        }
    }
    meta_payload: Dict[str, Any] = {}
    if status == "success" and _benchmark_config(config).get("route_to_meta_backtest", True):
        meta_payload = _run_meta_backtest_for_predictions(
            model_id,
            oof_predictions,
            test_predictions,
            dataset,
            config,
            model_dir,
            device or torch.device("cpu"),
        )
        metrics["backtest"] = meta_payload["metrics"]

    manifest = {
        "model_id": model_id,
        "kind": kind,
        "status": status,
        "reason": reason,
        "prediction_schema": _prediction_schema(dataset, test_predictions),
        "artifact_paths": {**paths, **meta_payload.get("artifact_paths", {})},
        "metrics": metrics,
        "meta_labeler": meta_payload.get("meta_labeler"),
        "manifest_path": model_dir / "manifest.json",
    }
    _write_json(model_dir / "manifest.json", manifest)
    return manifest


def _prediction_schema(dataset: Mapping[str, Any], frame: pd.DataFrame) -> Dict[str, Any]:
    columns = ["timestamp"]
    for horizon in dataset["horizon_names"]:
        columns.append(f"{horizon}_pred_return")
        if f"{horizon}_direction_prob" in frame.columns:
            columns.append(f"{horizon}_direction_prob")
        columns.extend([col for col in frame.columns if col.startswith(f"{horizon}_pred_return_q")])
    return {"required_columns": columns, "available_columns": list(frame.columns)}


def _artifact_groups(manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    artifacts = manifest.get("artifact_paths", {}) or {}
    prediction_paths = {key: artifacts[key] for key in ["oof_predictions", "test_predictions"] if key in artifacts}
    candidate_paths = {key: artifacts[key] for key in ["oof_candidates", "test_candidates"] if key in artifacts}
    trade_paths = {key: artifacts[key] for key in ["validation_trades", "test_trades"] if key in artifacts}
    return {"prediction_paths": prediction_paths, "candidate_paths": candidate_paths, "trade_paths": trade_paths}


def _infer_benchmark_run_id(run_dir: Path, run_id: Optional[str]) -> str:
    return str(run_id or run_dir.parent.name or run_dir.name)


def _log_benchmark_trackio_model(
    config: Mapping[str, Any],
    *,
    run_id: str,
    config_path: Optional[Path],
    manifest: Mapping[str, Any],
    smoke: bool,
) -> None:
    model_id = str(manifest.get("model_id", "unknown"))
    groups = _artifact_groups(manifest)
    artifacts = {"manifest": manifest.get("manifest_path")}
    if config_path is not None:
        artifacts["config"] = config_path
    artifacts.update(manifest.get("artifact_paths", {}) or {})
    if manifest.get("model_path") is not None:
        artifacts["model"] = manifest.get("model_path")
    artifacts = {key: value for key, value in artifacts.items() if value is not None}

    run_config: Dict[str, Any] = {
        "stage": "forecast_benchmark",
        "run_id": run_id,
        "model_id": model_id,
        "kind": manifest.get("kind"),
        "status": manifest.get("status", "success"),
        "smoke": smoke,
        "config_path": config_path,
        "manifest_path": manifest.get("manifest_path"),
        **groups,
    }
    if manifest.get("reason"):
        run_config["skip_reason"] = manifest.get("reason")

    log_trackio_run(
        config,
        name=f"benchmark/{run_id}/{model_id}",
        group="forecast_benchmark",
        run_config=run_config,
        metrics=manifest.get("metrics", {}) or {},
        artifacts=artifacts,
        status=str(manifest.get("status", "success")),
    )


def run_forecast_benchmark_from_dataset(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    run_dir: Path,
    *,
    smoke: bool = False,
    device: Optional[torch.device] = None,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    bench_cfg = _benchmark_config(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "split_manifest.json", {"note": "indices match v2 purged walk-forward split"})
    splits = purged_walk_forward_splits(int(dataset["X"].shape[0]), config.get("validation", {}))
    batch_size = int(bench_cfg.get("prediction_batch_size", 512))
    resolved_device = device or torch.device("cpu")

    model_ids = [str(item) for item in bench_cfg.get("models", DEFAULT_BENCHMARK_MODELS)]
    model_manifests: List[Dict[str, Any]] = []
    tracking_run_id = _infer_benchmark_run_id(run_dir, run_id)

    if "zero_return" in model_ids or "naive" in model_ids:
        model_manifests.append(
            _write_model_result(
                "zero_return",
                run_dir / "models" / "zero_return",
                zero_return_predictions(dataset, _concat_fold_indices(splits["folds"], "validation")),
                zero_return_predictions(dataset, splits["test"]),
                dataset,
                config,
                kind="baseline",
                device=resolved_device,
            )
        )

    if "momentum" in model_ids:
        model_manifests.append(
            _write_model_result(
                "momentum",
                run_dir / "models" / "momentum",
                momentum_predictions(dataset, _concat_fold_indices(splits["folds"], "validation")),
                momentum_predictions(dataset, splits["test"]),
                dataset,
                config,
                kind="baseline",
                device=resolved_device,
            )
        )

    if "lstm" in model_ids:
        lstm_dir = run_dir / "models" / "lstm"
        lstm_dir.mkdir(parents=True, exist_ok=True)
        oof_frames: List[pd.DataFrame] = []
        for fold in splits["folds"]:
            model, _ = train_base_model(dataset, fold["train"], config, resolved_device)
            fold_pred = predict_base_model(model, dataset, fold["validation"], resolved_device, batch_size=batch_size)
            fold_pred["fold"] = int(fold["fold"])
            oof_frames.append(fold_pred)
        oof_predictions = pd.concat(oof_frames, ignore_index=True).sort_values("timestamp")
        final_model, _ = train_base_model(dataset, splits["development"], config, resolved_device)
        torch.save(final_model.state_dict(), lstm_dir / "base_multi_horizon_lstm.pt")
        test_predictions = predict_base_model(final_model, dataset, splits["test"], resolved_device, batch_size=batch_size)
        manifest = _write_model_result(
            "lstm",
            lstm_dir,
            oof_predictions,
            test_predictions,
            dataset,
            config,
            kind="multi_horizon_lstm",
            device=resolved_device,
        )
        manifest["model_path"] = lstm_dir / "base_multi_horizon_lstm.pt"
        _write_json(lstm_dir / "manifest.json", manifest)
        model_manifests.append(manifest)

    timesfm_cfg = bench_cfg.get("timesfm", {}) or {}
    if "timesfm" in model_ids and bool(timesfm_cfg.get("enabled", True)):
        timesfm_dir = run_dir / "models" / "timesfm"
        try:
            adapter = TimesFMZeroShotAdapter(
                context_length=int(timesfm_cfg.get("context_length", 1024)),
                horizon_length=max(bars for _, bars in target_horizons(config)),
                checkpoint=str(timesfm_cfg.get("checkpoint", DEFAULT_TIMESFM_CHECKPOINT)),
                input_series=str(timesfm_cfg.get("input_series", "log_close")),
                batch_size=int(timesfm_cfg.get("batch_size", 16)),
                compile_kwargs=timesfm_cfg.get("compile_kwargs", {}) or {},
            )
            oof_predictions = timesfm_predictions(dataset, _concat_fold_indices(splits["folds"], "validation"), adapter)
            test_predictions = timesfm_predictions(dataset, splits["test"], adapter)
            model_manifests.append(
                _write_model_result(
                    "timesfm",
                    timesfm_dir,
                    oof_predictions,
                    test_predictions,
                    dataset,
                    config,
                    kind="timesfm_zero_shot",
                    device=resolved_device,
                )
            )
        except Exception as exc:
            timesfm_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "model_id": "timesfm",
                "kind": "timesfm_zero_shot",
                "status": "skipped",
                "reason": str(exc),
                "manifest_path": timesfm_dir / "manifest.json",
            }
            _write_json(timesfm_dir / "manifest.json", manifest)
            model_manifests.append(manifest)
    elif "timesfm" in model_ids:
        timesfm_dir = run_dir / "models" / "timesfm"
        timesfm_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_id": "timesfm",
            "kind": "timesfm_zero_shot",
            "status": "skipped",
            "reason": "TimesFM benchmark disabled in config",
            "manifest_path": timesfm_dir / "manifest.json",
        }
        _write_json(timesfm_dir / "manifest.json", manifest)
        model_manifests.append(manifest)

    for manifest in model_manifests:
        _log_benchmark_trackio_model(
            config,
            run_id=tracking_run_id,
            config_path=config_path,
            manifest=manifest,
            smoke=smoke,
        )

    summary = {
        "stage": "forecast_benchmark",
        "path": run_dir,
        "smoke": smoke,
        "models": model_manifests,
        "forecast_metrics_are_primary": True,
        "backtest_metrics_are_secondary": True,
        "manifest_path": run_dir / "manifest.json",
    }
    _write_json(run_dir / "manifest.json", summary)
    return summary


def _concat_fold_indices(folds: Sequence[Mapping[str, np.ndarray]], key: str) -> np.ndarray:
    if not folds:
        return np.array([], dtype=int)
    return np.concatenate([np.asarray(fold[key], dtype=int) for fold in folds])


def run_forecast_benchmark(
    config_path: Path,
    *,
    smoke: bool = False,
    device: str = "cuda:0",
    allow_cpu: bool = False,
    run_name: Optional[str] = None,
    artifact_root: Optional[Path] = None,
) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    config = apply_smoke_overrides(config) if smoke else config
    if run_name:
        config.setdefault("pipeline", {})["run_name"] = run_name
    if artifact_root:
        config.setdefault("pipeline", {})["artifact_root"] = str(artifact_root)

    pipeline = config.get("pipeline", {}) or {}
    run_id = _run_id(str(pipeline.get("run_name", "forecast_benchmark")))
    run_dir = Path(pipeline.get("artifact_root", "artifacts/runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_training_device(device, allow_cpu=allow_cpu)
    _write_yaml(run_dir / "resolved_config.yml", config)
    dataset = build_multi_horizon_lighter_dataset(config, smoke=smoke)
    summary = run_forecast_benchmark_from_dataset(
        config,
        dataset,
        run_dir / "forecast_benchmark",
        smoke=smoke,
        device=resolved_device,
        config_path=config_path,
        run_id=run_id,
    )
    report = {
        "run_id": run_id,
        "run_dir": run_dir,
        "config_path": config_path,
        "status": "success",
        "benchmark": summary,
        "report_json_path": run_dir / "report.json",
    }
    _write_json(run_dir / "report.json", report)
    return _json_ready(report)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run ETHPredict forecast benchmarks on the v2 multi-horizon split.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_forecast_benchmark(
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
