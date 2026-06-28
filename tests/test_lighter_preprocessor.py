from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features_all import DataPreprocessor


def _write_ohlcv(path: Path, start_ms: int, rows: int, close_offset: float = 0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "open_time": [start_ms + i * 3_600_000 for i in range(rows)],
            "open": [3000.0 + close_offset + i for i in range(rows)],
            "high": [3010.0 + close_offset + i for i in range(rows)],
            "low": [2990.0 + close_offset + i for i in range(rows)],
            "close": [3005.0 + close_offset + i for i in range(rows)],
            "volume": [10.0 + i for i in range(rows)],
            "close_time": [start_ms + i * 3_600_000 + 3_599_999 for i in range(rows)],
            "quote_asset_volume": [(3005.0 + close_offset + i) * (10.0 + i) for i in range(rows)],
            "number_of_trades": [0 for _ in range(rows)],
            "taker_buy_base_asset_volume": [0.0 for _ in range(rows)],
            "taker_buy_quote_asset_volume": [0.0 for _ in range(rows)],
            "ignore": [0 for _ in range(rows)],
        }
    )
    frame.to_csv(path, header=False, index=False)


def test_load_price_data_uses_lighter_raw_files_only(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_ohlcv(raw_dir / "ETHUSDT-1h-lighter-20260328-20260628.csv", 1774656000000, 30)
    _write_ohlcv(raw_dir / "ETHUSDT-1h-2025-04.csv", 1743465600000, 30, close_offset=10000)

    loaded = DataPreprocessor(data_dir=str(tmp_path), include_santiment=True).load_price_data()

    assert set(loaded) == {"1h"}
    assert len(loaded["1h"]) == 30
    assert loaded["1h"]["close"].max() < 4000


def test_lighter_ohlcv_base_dataset_has_no_legacy_source_features(tmp_path):
    _write_ohlcv(tmp_path / "raw" / "ETHUSDT-1h-lighter-20260328-20260628.csv", 1774656000000, 40)
    preprocessor = DataPreprocessor(data_dir=str(tmp_path))

    features_df, targets_df = preprocessor.get_base_dataset("1h")
    X, y = preprocessor.prepare_features("1h", sequence_length=5)

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
