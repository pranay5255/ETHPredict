from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.experiments.forecast_benchmark import (
    TimesFMZeroShotAdapter,
    run_forecast_benchmark,
    timesfm_predictions,
)
from src.experiments.meta_labeling_mvp import select_trials_with_trade_floor


def _write_5m_ohlcv(path: Path, rows: int = 96, step: float = 0.001) -> np.ndarray:
    interval_ms = 300_000
    start_ms = 1774656000000
    closes = 3000.0 * np.exp(np.arange(rows) * step)
    frame = pd.DataFrame(
        {
            "open_time": [start_ms + i * interval_ms for i in range(rows)],
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": [10.0 + i for i in range(rows)],
            "close_time": [start_ms + i * interval_ms + interval_ms - 1 for i in range(rows)],
            "quote_asset_volume": closes * (10.0 + np.arange(rows)),
            "number_of_trades": [0 for _ in range(rows)],
            "taker_buy_base_asset_volume": [0.0 for _ in range(rows)],
            "taker_buy_quote_asset_volume": [0.0 for _ in range(rows)],
            "ignore": [0 for _ in range(rows)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, header=False, index=False)
    return closes


def _v2_config(data_dir: Path, artifact_root: Path) -> dict:
    return {
        "version": 2,
        "experiment": {"id": "pytest_benchmark", "seed": 11, "trials": 1},
        "data": {"dir": str(data_dir), "granularity": "5m", "sources": ["lighter"]},
        "targets": {"horizons": {"next_5m": {"bars": 1}, "next_hour": {"bars": 3}}},
        "validation": {"method": "purged_walk_forward", "folds": 2, "test_fraction": 0.2, "purge_bars": 2, "embargo_bars": 2},
        "labels": {
            "meta_triple_barrier": {
                "profit_kappa": 1.0,
                "stop_kappa": 1.0,
                "vertical_barriers": {"next_5m": 1, "next_hour": 3},
                "volatility_window": 8,
                "min_edge_bps": 0.0,
                "cost_mode": "configured",
            }
        },
        "model": {
            "base": {"type": "lstm", "hidden_size": 4, "num_layers": 1, "dropout": 0.0},
            "meta_labeler": {"type": "mlp", "hidden_size": 4, "num_layers": 1, "dropout": 0.0, "enabled": True},
        },
        "training": {"sequence_length": 8, "epochs": 1, "batch_size": 8, "learning_rate": 0.001, "loss_weights": {"return": 1.0, "direction": 0.1}},
        "costs": {"fee_bps": 1.0, "spread_bps": 1.0, "slippage_bps": 0.0, "funding_bps_per_hour": 0.0},
        "alpha_backtest": {"initial_capital": 10000.0, "horizon": "best", "meta_threshold": 0.5, "edge_threshold_bps": 0.0, "position_notional": 1000.0},
        "pipeline": {"artifact_root": str(artifact_root), "run_name": "pytest_benchmark", "selection_metric": "metrics.validation.net_pnl", "selection_mode": "max"},
        "search": {"mode": "grid", "max_trials": 1, "spaces": {}},
        "benchmark": {"enabled": True, "models": ["zero_return", "momentum"], "route_to_meta_backtest": True},
        "smoke": {"max_rows": 96, "max_trials": 1, "folds": 2, "sequence_length": 8, "epochs": 1, "hidden_size": 4, "batch_size": 8},
    }


def _synthetic_dataset() -> dict:
    close = 100.0 * np.exp(np.arange(40) * 0.001)
    sample_indices = np.arange(12, 20)
    timestamps = pd.date_range("2026-01-01", periods=len(sample_indices), freq="5min")
    samples = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sample_index": sample_indices,
            "close": close[sample_indices],
            "open": close[sample_indices],
            "high": close[sample_indices],
            "low": close[sample_indices],
            "volume": 1.0,
            "realized_vol": 0.001,
            "vol_regime": 0.0,
        }
    )
    y_ret = []
    for idx in sample_indices:
        y_ret.append([np.log(close[idx + 1] / close[idx]), np.log(close[idx + 12] / close[idx])])
    return {
        "samples": samples,
        "price_path": pd.DataFrame({"close": close}),
        "y_ret": torch.tensor(y_ret, dtype=torch.float32),
        "y_dir": torch.tensor(np.asarray(y_ret) > 0, dtype=torch.float32),
        "horizon_names": ["next_5m", "next_hour"],
        "horizon_bars": {"next_5m": 1, "next_hour": 12},
        "granularity": "5m",
    }


class FakeTimesFM:
    def forecast(self, *args, **kwargs):
        inputs = kwargs.get("inputs") or args[0]
        horizon = int(kwargs.get("horizon", 12))
        points = []
        quantiles = []
        for series in inputs:
            step = float(series[-1] - series[-2]) if len(series) > 1 else 0.0
            point = series[-1] + step * np.arange(1, horizon + 1)
            points.append(point)
            quantiles.append(np.stack([point - 0.001, point, point + 0.001], axis=1))
        return np.asarray(points), np.asarray(quantiles)


def test_timesfm_adapter_outputs_standard_schema_from_synthetic_close_series():
    dataset = _synthetic_dataset()
    adapter = TimesFMZeroShotAdapter(context_length=8, horizon_length=12, model=FakeTimesFM(), input_series="log_close")

    frame = timesfm_predictions(dataset, np.array([0, 1, 2]), adapter)

    assert {"timestamp", "next_5m_pred_return", "next_hour_pred_return"}.issubset(frame.columns)
    assert {"next_5m_pred_return_q10", "next_5m_pred_return_q50", "next_5m_pred_return_q90"}.issubset(frame.columns)
    assert np.allclose(frame["next_5m_pred_return"], 0.001, atol=1e-8)
    assert np.allclose(frame["next_hour_pred_return"], 0.012, atol=1e-8)


def test_no_trade_trial_is_reported_as_abstention_not_best_trading_result():
    trials = [
        {
            "trial_id": "no_trade",
            "metrics": {"validation": {"net_pnl": 0.0, "trades": 0.0}, "test": {"net_pnl": 0.0, "trades": 0.0}},
        },
        {
            "trial_id": "trading_loss",
            "metrics": {"validation": {"net_pnl": -1.0, "trades": 2.0}, "test": {"net_pnl": -1.0, "trades": 2.0}},
        },
    ]
    config = {"pipeline": {"selection_metric": "metrics.validation.net_pnl", "selection_mode": "max", "min_validation_trades": 1}}

    raw_best, best_trading, selection = select_trials_with_trade_floor(trials, config)

    assert raw_best["trial_id"] == "no_trade"
    assert raw_best["selection_status"] == "abstention"
    assert best_trading["trial_id"] == "trading_loss"
    assert selection["best_trading_trial_id"] == "trading_loss"


def test_forecast_benchmark_smoke_writes_forecast_and_backtest_metrics(tmp_path):
    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv")
    config_path = tmp_path / "benchmark.yml"
    config_path.write_text(yaml.safe_dump(_v2_config(data_dir, tmp_path / "runs")), encoding="utf-8")

    result = run_forecast_benchmark(config_path, smoke=True, device="cpu", allow_cpu=True)

    models = {item["model_id"]: item for item in result["benchmark"]["models"]}
    assert {"zero_return", "momentum"} <= set(models)
    assert models["zero_return"]["status"] == "success"
    assert "forecast" in models["momentum"]["metrics"]
    assert "backtest" in models["momentum"]["metrics"]
    assert "next_5m_pred_return" in models["momentum"]["prediction_schema"]["required_columns"]
