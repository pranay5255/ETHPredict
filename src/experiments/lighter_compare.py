from __future__ import annotations

import argparse
import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data.features_all import DEFAULT_GRANULARITY, DataPreprocessor, bars_for_duration
from src.features.labeling import create_labels, sample_weights_from_labels
from src.market_maker.glft import GLFTParams, GLFTQuoteCalculator
from src.market_maker.inventory import InventoryBook
from src.models.model import ConfidenceGRU, MetaMLP, PriceLSTM
from src.simulator.backtest import BacktestParams, GLFTBacktester
from src.training.devices import resolve_training_device
from src.training.trainer import compute_metrics, hierarchical_training_pipeline


DEFAULT_SEQUENCE_LENGTH = bars_for_duration(DEFAULT_GRANULARITY, hours=24)
DEFAULT_NEXT_HOUR_HORIZON = bars_for_duration(DEFAULT_GRANULARITY, hours=1)

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {"dir": "data", "granularity": DEFAULT_GRANULARITY},
    "targets": ["triple_barrier", "next_hour_return"],
    "labels": {
        "kappa": 2.0,
        "timeout": DEFAULT_SEQUENCE_LENGTH,
        "volatility_window": DEFAULT_SEQUENCE_LENGTH,
        "next_hour_horizon": DEFAULT_NEXT_HOUR_HORIZON,
    },
    "training": {
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_size": 16,
        "epochs": 3,
        "learning_rate": 0.001,
        "neural_trials_per_target": 2,
    },
    "smoke": {
        "max_rows": 128,
        "neural_trials_per_target": 1,
        "epochs": 1,
        "hidden_size": 8,
        "batch_size": 8,
        "baseline_points": 80,
    },
    "glft": {
        "initial_capital": 100_000.0,
        "gamma": 0.5,
        "kappa": 0.1,
        "dt": 1.0 / DEFAULT_SEQUENCE_LENGTH,
        "max_inventory": 10.0,
        "min_spread": 5.0,
        "max_drawdown": 0.2,
        "seed": 42,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_experiment_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_update(DEFAULT_CONFIG, loaded)


def _training_config(config: Dict[str, Any], smoke: bool) -> Dict[str, Any]:
    training = dict(config["training"])
    if smoke:
        smoke_cfg = config.get("smoke", {})
        for key in ["neural_trials_per_target", "epochs", "hidden_size", "batch_size"]:
            if key in smoke_cfg:
                training[key] = smoke_cfg[key]
    return training


def _normalise_features(features_df: pd.DataFrame) -> torch.Tensor:
    features = torch.tensor(features_df.values, dtype=torch.float32)
    mean = features.mean(dim=0)
    std = features.std(dim=0) + 1e-8
    return (features - mean) / std


def build_lighter_dataset(config: Dict[str, Any], target: str, *, smoke: bool) -> Dict[str, Any]:
    data_cfg = config["data"]
    granularity = data_cfg.get("granularity", DEFAULT_GRANULARITY)
    training_cfg = _training_config(config, smoke)
    sequence_length = int(training_cfg["sequence_length"])
    label_cfg = config["labels"]
    timeout_bars = int(label_cfg.get("timeout", bars_for_duration(granularity, hours=24)))
    volatility_window = int(label_cfg.get("volatility_window", timeout_bars))
    next_hour_horizon = int(label_cfg.get("next_hour_horizon", bars_for_duration(granularity, hours=1)))

    preprocessor = DataPreprocessor(data_dir=data_cfg["dir"], granularities=[granularity])
    features_df, targets_df = preprocessor.get_base_dataset(granularity=granularity)

    max_rows = int(config.get("smoke", {}).get("max_rows", 0)) if smoke else 0
    if max_rows:
        label_lookahead = timeout_bars if target == "triple_barrier" else next_hour_horizon
        min_rows = sequence_length + label_lookahead + 2
        max_rows = max(max_rows, min_rows)
        features_df = features_df.tail(max_rows)
        targets_df = targets_df.tail(max_rows)

    prices = targets_df["close"].astype(float)
    if target == "triple_barrier":
        labels = create_labels(
            pd.DataFrame({"close": prices}, index=prices.index),
            price_col="close",
            kappa=float(label_cfg["kappa"]),
            timeout=timeout_bars,
            volatility_window=volatility_window,
        )
        y_ret = labels["y_ret"].astype(float)
        y_dir = labels["y_dir"].astype(int)
        weights = sample_weights_from_labels(
            labels["y_dir"],
            labels["hit_times"],
            labels["returns"],
            volatility_window=volatility_window,
        )
    elif target == "next_hour_return":
        horizon = next_hour_horizon
        y_ret = np.log(prices.shift(-horizon) / prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_dir = np.sign(y_ret).astype(int)
        weights = pd.Series(1.0 / max(len(y_ret), 1), index=y_ret.index)
    else:
        raise ValueError(f"Unknown target: {target}")

    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(weights.sum()) <= 0.0:
        weights = pd.Series(1.0 / max(len(weights), 1), index=weights.index)

    scaled_features = _normalise_features(features_df)
    X, ret_values, dir_values, weight_values, close_values, timestamps = [], [], [], [], [], []
    for idx in range(len(features_df) - sequence_length):
        label_idx = idx + sequence_length
        X.append(scaled_features[idx:label_idx])
        ret_values.append(float(y_ret.iloc[label_idx]))
        dir_values.append(int(y_dir.iloc[label_idx]))
        weight_values.append(float(weights.iloc[label_idx]))
        close_values.append(float(prices.iloc[label_idx]))
        timestamps.append(prices.index[label_idx])

    if not X:
        raise ValueError(f"Not enough rows for sequence_length={sequence_length} and target={target}")

    return {
        "target": target,
        "X": torch.stack(X),
        "y_ret": torch.tensor(ret_values, dtype=torch.float32).unsqueeze(1),
        "y_dir": torch.tensor(dir_values, dtype=torch.long),
        "sample_weights": torch.tensor(weight_values, dtype=torch.float32),
        "close": pd.Series(close_values, index=pd.Index(timestamps, name="timestamp")),
        "input_size": int(scaled_features.shape[1]),
        "granularity": granularity,
        "sequence_length": sequence_length,
        "timeout_bars": timeout_bars,
        "next_hour_horizon_bars": next_hour_horizon,
    }


def run_stack_forward_backward(input_size: int, sequence_length: int, device: torch.device) -> Dict[str, Any]:
    torch.manual_seed(42)
    batch_size = 4
    hidden_size = 8
    X = torch.randn(batch_size, sequence_length, input_size, device=device)
    price_model = PriceLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, dropout=0.0).to(device)
    meta_model = MetaMLP(input_size=input_size + 1, hidden_size=hidden_size, num_layers=1, dropout=0.0).to(device)
    confidence_model = ConfidenceGRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, dropout=0.0).to(device)

    base_pred = price_model(X)
    refined_pred = meta_model(base_pred, X[:, -1, :])
    confidence = confidence_model(X)
    loss = base_pred.pow(2).mean() + refined_pred.pow(2).mean() + confidence.mean()
    loss.backward()

    return {
        "device": str(device),
        "loss": float(loss.detach().cpu()),
        "models": ["PriceLSTM", "MetaMLP", "ConfidenceGRU"],
    }


def run_arima_sarimax_smoke(values: np.ndarray, *, max_points: int = 80) -> Dict[str, Dict[str, float]]:
    series = pd.Series(np.asarray(values, dtype=float).reshape(-1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    series = series.tail(max_points).reset_index(drop=True)
    if len(series) < 12:
        raise ValueError("Need at least 12 target observations for ARIMA/SARIMAX smoke tests")

    results: Dict[str, Dict[str, float]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima_fit = ARIMA(series, order=(1, 0, 0)).fit()
        arima_forecast = float(arima_fit.forecast(1).iloc[0])
        results["arima"] = {"forecast": arima_forecast, "aic": float(arima_fit.aic)}

        sarimax_fit = SARIMAX(
            series,
            order=(1, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        sarimax_forecast = float(sarimax_fit.forecast(1).iloc[0])
        results["sarimax"] = {"forecast": sarimax_forecast, "aic": float(sarimax_fit.aic)}
    return results


def _glft_metrics(config: Dict[str, Any], closes: pd.Series, predicted_returns: np.ndarray) -> Dict[str, float]:
    if len(closes) < 3:
        return {"num_trades": 0.0}

    glft_cfg = config["glft"]
    clipped_returns = np.asarray(predicted_returns, dtype=float).reshape(-1)
    predicted_prices = closes * np.exp(np.clip(clipped_returns, -0.05, 0.05))
    granularity = config.get("data", {}).get("granularity", DEFAULT_GRANULARITY)
    one_hour_bars = bars_for_duration(granularity, hours=1)
    volatility = closes.pct_change().rolling(one_hour_bars, min_periods=2).std().fillna(closes.pct_change().std() or 0.001)

    params = GLFTParams(
        gamma=float(glft_cfg["gamma"]),
        kappa=float(glft_cfg["kappa"]),
        sigma=float(max(volatility.mean(), 1e-6)),
        dt=float(glft_cfg["dt"]),
        max_inventory=float(glft_cfg["max_inventory"]),
        min_spread=float(glft_cfg["min_spread"]),
    )
    start = pd.to_datetime(closes.index[0]).to_pydatetime()
    end = pd.to_datetime(closes.index[-1]).to_pydatetime()
    backtester = GLFTBacktester(
        BacktestParams(
            start_date=start,
            end_date=end,
            initial_capital=float(glft_cfg["initial_capital"]),
            seed=int(glft_cfg["seed"]),
            gamma=float(glft_cfg["gamma"]),
            inventory_limit=float(glft_cfg["max_inventory"]),
            quote_spread=float(glft_cfg["min_spread"]),
        ),
        GLFTQuoteCalculator(params),
        InventoryBook(max_position=float(glft_cfg["max_inventory"]), max_drawdown=float(glft_cfg["max_drawdown"])),
    )
    result = backtester.run(pd.DataFrame({"price": closes}), predicted_prices, volatility=volatility)
    metrics: Dict[str, float] = {}
    for key, value in result.metrics.items():
        if not np.isscalar(value):
            continue
        scalar = float(value)
        metrics[key] = scalar if np.isfinite(scalar) else 0.0
    return metrics


def run_neural_trial(
    dataset: Dict[str, Any],
    config: Dict[str, Any],
    training_cfg: Dict[str, Any],
    *,
    device: torch.device,
    allow_cpu: bool,
    trial_index: int,
) -> Dict[str, Any]:
    torch.manual_seed(42 + trial_index)
    results = hierarchical_training_pipeline(
        dataset["X"],
        dataset["y_ret"],
        dataset["y_dir"],
        dataset["sample_weights"],
        input_size=dataset["input_size"],
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
    model.eval()
    with torch.no_grad():
        predictions, confidence = model(dataset["X"].to(device), return_confidence=True)

    prediction_metrics = compute_metrics(dataset["y_ret"], predictions.cpu(), "regression")
    confidence_targets = (dataset["y_dir"] != 0).float().unsqueeze(1)
    confidence_metrics = compute_metrics(confidence_targets, confidence.cpu(), "classification")
    glft_metrics = _glft_metrics(config, dataset["close"], predictions.cpu().numpy().reshape(-1))

    return {
        "trial": trial_index,
        "prediction": {key: float(value) for key, value in prediction_metrics.items()},
        "confidence": {key: float(value) for key, value in confidence_metrics.items()},
        "glft": glft_metrics,
    }


def run_experiment(
    config_path: Path,
    *,
    smoke: bool = False,
    device: str = "cuda:0",
    allow_cpu: bool = False,
) -> Dict[str, Any]:
    config = load_experiment_config(config_path)
    training_cfg = _training_config(config, smoke)
    resolved_device = resolve_training_device(device, allow_cpu=allow_cpu)
    targets: List[str] = list(config.get("targets", DEFAULT_CONFIG["targets"]))

    output: Dict[str, Any] = {
        "config": str(config_path),
        "smoke": smoke,
        "device": str(resolved_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "targets": {},
    }

    stack_checked = False
    for target in targets:
        dataset = build_lighter_dataset(config, target, smoke=smoke)
        if not stack_checked:
            output["stack_smoke"] = run_stack_forward_backward(
                dataset["input_size"],
                int(training_cfg["sequence_length"]),
                resolved_device,
            )
            stack_checked = True

        trial_count = int(training_cfg["neural_trials_per_target"])
        neural_trials = [
            run_neural_trial(
                dataset,
                config,
                training_cfg,
                device=resolved_device,
                allow_cpu=allow_cpu,
                trial_index=trial_index,
            )
            for trial_index in range(trial_count)
        ]
        finalists = sorted(
            neural_trials,
            key=lambda item: (item["prediction"]["mae"], -item["glft"].get("net_pnl", 0.0)),
        )
        output["targets"][target] = {
            "samples": int(dataset["X"].shape[0]),
            "granularity": dataset["granularity"],
            "sequence_length": dataset["sequence_length"],
            "timeout_bars": dataset["timeout_bars"],
            "next_hour_horizon_bars": dataset["next_hour_horizon_bars"],
            "neural_trials": neural_trials,
            "best_trial": finalists[0],
            "baselines": run_arima_sarimax_smoke(
                dataset["y_ret"].numpy(),
                max_points=int(config.get("smoke", {}).get("baseline_points", 80)) if smoke else 160,
            ),
        }

    return output


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare Lighter-only neural and CPU time-series baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/lighter_experiments.yml"))
    parser.add_argument("--smoke", action="store_true", help="Run the tiny GPU/CPU smoke budget.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for neural trials. Defaults to cuda:0.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow neural trials to run on CPU.")
    args = parser.parse_args(argv)

    result = run_experiment(args.config, smoke=args.smoke, device=args.device, allow_cpu=args.allow_cpu)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
