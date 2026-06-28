import pandas as pd

from src.data.lighter_client import (
    build_candle_windows,
    candles_to_model_ohlcv,
    normalize_candles,
    normalize_funding_rates,
    normalize_fundings,
    to_epoch_ms,
)


def test_build_candle_windows_caps_count_back():
    start = to_epoch_ms("2025-01-17")
    end = start + 501 * 60_000

    windows = build_candle_windows(start, end, "1m", count_back=500)

    assert len(windows) == 2
    assert windows[0].count_back == 500
    assert windows[1].count_back == 1


def test_normalize_candles_and_convert_to_model_csv_shape():
    payload = {
        "code": 200,
        "r": "1h",
        "c": [
            {
                "t": 1737072000000,
                "o": 3300.0,
                "h": 3310.0,
                "l": 3290.0,
                "c": 3305.0,
                "v": 12.5,
                "V": 41312.5,
                "i": 123,
            }
        ],
    }

    candles = normalize_candles(payload)
    out = candles_to_model_ohlcv(candles, "1h")

    assert list(out.columns) == [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    assert out.loc[0, "open_time"] == 1737072000000
    assert out.loc[0, "close_time"] == 1737075599999
    assert out.loc[0, "close"] == 3305.0


def test_normalize_fundings_handles_seconds_and_numeric_strings():
    payload = {
        "code": 200,
        "resolution": "1h",
        "fundings": [{"timestamp": 1737072000, "value": "1.25", "rate": "0.0001", "direction": "long"}],
    }

    df = normalize_fundings(payload)

    assert df.loc[0, "timestamp_ms"] == 1737072000000
    assert df.loc[0, "value"] == 1.25
    assert df.loc[0, "rate"] == 0.0001


def test_normalize_funding_rates_returns_expected_columns():
    payload = {
        "code": 200,
        "funding_rates": [
            {"market_id": 0, "exchange": "lighter", "symbol": "ETH", "rate": 0.000024},
        ],
    }

    df = normalize_funding_rates(payload)

    assert list(df.columns) == ["market_id", "exchange", "symbol", "rate"]
    assert df.loc[0, "symbol"] == "ETH"
    assert pd.api.types.is_float_dtype(df["rate"])
