"""Vendored source feature-family and duration contracts; no feature generation."""
from typing import Dict, List

RESOLUTION_TO_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}
MS_PER_HOUR = 60 * 60 * 1000

FEATURE_FAMILY_COLUMNS: Dict[str, List[str]] = {
    "ohlcv": ["open", "high", "low", "close", "volume", "quote_asset_volume"],
    "returns": ["price_return", "log_return", "volume_change", "quote_volume_change", "dollar_volume", "high_low_range", "close_open_return"],
    "realized_volatility": ["return_vol_1bar", "return_vol_24h", "return_vol_7d", "parkinson_vol"],
    "volume": ["volume_zscore_24h"],
    "fracdiff": ["fracdiff_close"],
    "entropy": ["return_entropy_24h", "return_entropy_7d"],
    "event_flags": ["cusum_flag"],
    "structural_break": ["sadf_flag"],
    "volatility_regime": ["vol_regime"],
}

def bars_for_duration(granularity: str, *, hours: float) -> int:
    """Return the number of bars needed to preserve a wall-clock duration."""

    if granularity not in RESOLUTION_TO_MS:
        raise ValueError(f"Unsupported Lighter candle granularity: {granularity}")
    bars = round((hours * MS_PER_HOUR) / RESOLUTION_TO_MS[granularity])
    return max(1, int(bars))
