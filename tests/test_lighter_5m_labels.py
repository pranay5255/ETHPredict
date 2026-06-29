from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.lighter_compare import build_lighter_dataset
from src.features.labeling import create_labels


def _write_exp_ohlcv(path: Path, rows: int, start_ms: int = 1774656000000, step: float = 0.01):
    interval_ms = 300_000
    closes = np.exp(np.arange(rows) * step) * 3000.0
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


def test_next_hour_return_uses_12_5m_bars(tmp_path):
    closes = _write_exp_ohlcv(tmp_path / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", rows=40)
    config = {
        "data": {"dir": str(tmp_path), "granularity": "5m"},
        "labels": {"kappa": 2.0, "timeout": 288, "volatility_window": 288, "next_hour_horizon": 12},
        "training": {"sequence_length": 2},
        "smoke": {},
    }

    dataset = build_lighter_dataset(config, "next_hour_return", smoke=False)

    expected = np.log(closes[14] / closes[2])
    assert dataset["next_hour_horizon_bars"] == 12
    assert np.isclose(float(dataset["y_ret"][0]), expected)


def test_triple_barrier_timeout_uses_288_5m_bars():
    prices = pd.Series(np.full(300, 3000.0))
    labels = create_labels(
        pd.DataFrame({"close": prices}),
        price_col="close",
        timeout=288,
        volatility_window=288,
    )

    assert int(labels["hit_times"].iloc[0]) == 288
    assert labels["y_ret"].iloc[0] == 0.0
    assert labels["y_dir"].iloc[0] == 0
