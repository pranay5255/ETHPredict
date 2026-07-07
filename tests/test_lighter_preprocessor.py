from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features_all import DataPreprocessor, bars_for_duration, join_side_data_asof, validate_side_data_coverage


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


def test_feature_include_controls_columns_and_fracdiff_mode(tmp_path):
    _write_ohlcv(tmp_path / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", 1774656000000, 80)
    preprocessor = DataPreprocessor(
        data_dir=str(tmp_path),
        feature_config={"include": ["ohlcv", "fracdiff"], "frac_diff_order": 0.4, "frac_diff_mode": "fixed_width"},
    )

    features_df, _ = preprocessor.get_base_dataset("5m")
    manifest = preprocessor.feature_manifest()

    assert list(features_df.columns) == ["open", "high", "low", "close", "volume", "quote_asset_volume", "fracdiff_close"]
    assert manifest["families"] == ["ohlcv", "fracdiff"]
    assert manifest["fracdiff"]["mode"] == "fixed_width"
    assert manifest["fracdiff"]["order"] == 0.4
    assert manifest["fracdiff"]["valid_rows"] > 0
    assert "AFML-inspired" in manifest["fracdiff"]["implementation_note"]
    assert "sadf_flag" in manifest["transform_notes"]


def test_volume_bar_construction_records_coverage(tmp_path):
    _write_ohlcv(tmp_path / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", 1774656000000, 20)
    preprocessor = DataPreprocessor(data_dir=str(tmp_path), bar_config={"type": "volume", "threshold_volume": 35.0, "min_rows": 2})

    loaded = preprocessor.load_price_data()["5m"]
    report = preprocessor.bar_manifest()["coverage"]["5m"]

    assert 2 <= len(loaded) < 20
    assert report["bar_type"] == "volume"
    assert report["source_rows"] == 20
    assert report["rows"] == len(loaded)


def test_dollar_bar_construction_deduplicates_and_reports_gaps(tmp_path):
    raw_path = tmp_path / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv"
    _write_ohlcv(raw_path, 1774656000000, 12)
    raw_text = raw_path.read_text(encoding="utf-8")
    raw_path.write_text(raw_text + raw_text.splitlines()[3] + "\n", encoding="utf-8")
    preprocessor = DataPreprocessor(data_dir=str(tmp_path), bar_config={"type": "dollar", "threshold_usd": 100000.0, "min_rows": 2})

    loaded = preprocessor.load_price_data()["5m"]
    report = preprocessor.bar_manifest()["coverage"]["5m"]

    assert len(loaded) < 13
    assert report["bar_type"] == "dollar"
    assert report["duplicate_timestamps"] == 0
    assert report["max_interval_seconds"] >= report["median_interval_seconds"]


def test_side_data_asof_join_reports_and_enforces_coverage():
    base = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=4, freq="5min"), "close": [1.0, 2.0, 3.0, 4.0]})
    side = pd.DataFrame({"timestamp": base["timestamp"].iloc[:2], "funding_rate": [0.1, 0.2]})

    joined, report = join_side_data_asof(base, side, columns=["funding_rate"], tolerance=pd.Timedelta(minutes=1), prefix="funding_")

    assert "funding_funding_rate" in joined.columns
    assert report["coverage"]["funding_funding_rate"] == 0.5
    validate_side_data_coverage(report, min_coverage=0.5)
    try:
        validate_side_data_coverage(report, min_coverage=0.75)
    except ValueError as exc:
        assert "below required threshold" in str(exc)
    else:
        raise AssertionError("expected side-data coverage gate to fail")
