from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features_all import DataPreprocessor, bars_for_duration


def _write_ohlcv(path: Path, start_ms: int, rows: int, close_offset: float = 0.0, interval_ms: int = 300_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "open_time": [start_ms + i * interval_ms for i in range(rows)],
            "open": [3000.0 + close_offset + i for i in range(rows)],
            "high": [3010.0 + close_offset + i for i in range(rows)],
            "low": [2990.0 + close_offset + i for i in range(rows)],
            "close": [3005.0 + close_offset + i for i in range(rows)],
            "volume": [10.0 + i for i in range(rows)],
            "close_time": [start_ms + i * interval_ms + interval_ms - 1 for i in range(rows)],
            "quote_asset_volume": [(3005.0 + close_offset + i) * (10.0 + i) for i in range(rows)],
            "number_of_trades": [0 for _ in range(rows)],
            "taker_buy_base_asset_volume": [0.0 for _ in range(rows)],
            "taker_buy_quote_asset_volume": [0.0 for _ in range(rows)],
            "ignore": [0 for _ in range(rows)],
        }
    )
    frame.to_csv(path, header=False, index=False)


def test_load_price_data_defaults_to_5m_lighter_raw_files_only(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_ohlcv(raw_dir / "ETHUSDT-5m-lighter-20260328-20260628.csv", 1774656000000, 30)
    _write_ohlcv(raw_dir / "ETHUSDT-1h-lighter-20260328-20260628.csv", 1774656000000, 30, interval_ms=3_600_000)
    _write_ohlcv(raw_dir / "ETHUSDT-5m-2025-04.csv", 1743465600000, 30, close_offset=10000)

    loaded = DataPreprocessor(data_dir=str(tmp_path), include_santiment=True).load_price_data()

    assert set(loaded) == {"5m"}
    assert len(loaded["5m"]) == 30
    assert loaded["5m"]["close"].max() < 4000


def test_lighter_ohlcv_base_dataset_has_no_legacy_source_features(tmp_path):
    _write_ohlcv(tmp_path / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", 1774656000000, 40)
    preprocessor = DataPreprocessor(data_dir=str(tmp_path))

    features_df, targets_df = preprocessor.get_base_dataset()
    X, y = preprocessor.prepare_features("5m", sequence_length=5)

    legacy_columns = {
        "tvl_usd",
        "tvl_change",
        "price_tvl_ratio",
        "volume_tvl_ratio",
        "address_growth",
        "social_dominance_change",
        "mcap_tvl_ratio",
    }
    assert legacy_columns.isdisjoint(features_df.columns)
    assert list(targets_df.columns) == ["close", "volume"]
    assert X.shape == (35, 5, len(preprocessor.get_feature_cols()))
    assert y.shape == (35, 1)
    assert np.isfinite(features_df.to_numpy()).all()
    assert np.isfinite(X).all()


def test_5m_feature_windows_preserve_wall_clock_durations(tmp_path):
    _write_ohlcv(tmp_path / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", 1774656000000, 300)
    preprocessor = DataPreprocessor(data_dir=str(tmp_path))

    windows = preprocessor.feature_window_bars("5m")
    features_df, _ = preprocessor.get_base_dataset("5m")

    assert bars_for_duration("5m", hours=24) == 288
    assert bars_for_duration("5m", hours=24 * 7) == 2016
    assert windows["twenty_four_hours"] == 288
    assert windows["seven_days"] == 2016
    assert {"return_vol_24h", "return_vol_7d", "volume_zscore_24h"}.issubset(features_df.columns)
    assert "return_vol_24" not in features_df.columns
