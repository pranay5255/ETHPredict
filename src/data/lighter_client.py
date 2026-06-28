"""Public market-data client for Lighter.

The training pipeline is CSV-first, so this module normalizes public Lighter
REST responses into pandas data frames and model-compatible OHLCV CSV rows.
Authenticated trading and account APIs should live in a separate execution
module.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd
import requests


DEFAULT_BASE_URL = "https://mainnet.zklighter.elliot.ai"
MAX_CANDLES_PER_CALL = 500
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
FUNDING_RESOLUTIONS = {"1h", "1d"}

TimestampLike = Union[str, int, float, datetime, pd.Timestamp]


class LighterApiError(RuntimeError):
    """Raised when Lighter returns a non-successful REST response."""


@dataclass(frozen=True)
class CandleWindow:
    """A single Lighter candle request window."""

    start_timestamp: int
    end_timestamp: int
    count_back: int


def to_epoch_ms(value: TimestampLike) -> int:
    """Convert a date/time value to Unix milliseconds.

    Numeric values under 100,000,000,000 are treated as seconds; larger numeric
    values are treated as milliseconds.
    """

    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, (int, float)):
        numeric = int(value)
        return numeric * 1000 if numeric < 100_000_000_000 else numeric
    else:
        ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return int(ts.timestamp() * 1000)


def from_epoch_ms(value: Union[int, float]) -> pd.Timestamp:
    """Convert seconds or milliseconds since epoch to a UTC timestamp."""

    numeric = int(value)
    if numeric < 100_000_000_000:
        numeric *= 1000
    return pd.to_datetime(numeric, unit="ms", utc=True)


def build_candle_windows(
    start_timestamp: int,
    end_timestamp: int,
    resolution: str,
    count_back: int = MAX_CANDLES_PER_CALL,
) -> List[CandleWindow]:
    """Build API windows respecting Lighter's 500-candle response cap."""

    if resolution not in RESOLUTION_TO_MS:
        raise ValueError(f"Unsupported Lighter candle resolution: {resolution}")
    if count_back < 1 or count_back > MAX_CANDLES_PER_CALL:
        raise ValueError("count_back must be between 1 and 500")
    if end_timestamp <= start_timestamp:
        raise ValueError("end_timestamp must be after start_timestamp")

    interval_ms = RESOLUTION_TO_MS[resolution]
    max_span_ms = interval_ms * count_back
    windows: List[CandleWindow] = []
    current = start_timestamp

    while current < end_timestamp:
        window_end = min(end_timestamp, current + max_span_ms)
        candles_in_window = max(1, math.ceil((window_end - current) / interval_ms))
        windows.append(
            CandleWindow(
                start_timestamp=current,
                end_timestamp=window_end,
                count_back=min(count_back, candles_in_window),
            )
        )
        current = window_end

    return windows


def _drop_none(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


def _records(payload: Mapping[str, Any], key: str) -> Sequence[Mapping[str, Any]]:
    records = payload.get(key, [])
    if records is None:
        return []
    return records


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_candles(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Normalize `/api/v1/candles` or mark price candle payloads."""

    rows = list(_records(payload, "c"))
    columns = [
        "timestamp_ms",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "last_trade_id",
        "sample_count",
        "open_raw",
        "high_raw",
        "low_raw",
        "close_raw",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    rename_map = {
        "t": "timestamp_ms",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "V": "quote_asset_volume",
        "i": "last_trade_id",
        "sc": "sample_count",
        "O": "open_raw",
        "H": "high_raw",
        "L": "low_raw",
        "C": "close_raw",
    }
    df = df.rename(columns=rename_map)

    for col in columns:
        if col not in df.columns and col != "timestamp":
            df[col] = 0

    df["timestamp_ms"] = df["timestamp_ms"].map(to_epoch_ms).astype("int64")
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert(None)
    numeric_cols = [col for col in columns if col not in {"timestamp"}]
    for col in numeric_cols:
        df[col] = _to_numeric(df[col]).fillna(0)

    df = df[columns].drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")
    return df.reset_index(drop=True)


def normalize_fundings(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = list(_records(payload, "fundings"))
    columns = ["timestamp_ms", "timestamp", "value", "rate", "direction"]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["timestamp_ms"] = df["timestamp"].map(to_epoch_ms).astype("int64")
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert(None)
    for col in ["value", "rate"]:
        df[col] = _to_numeric(df[col]).fillna(0.0)
    if "direction" not in df.columns:
        df["direction"] = ""
    return df[columns].drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)


def normalize_funding_rates(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = list(_records(payload, "funding_rates"))
    columns = ["market_id", "exchange", "symbol", "rate"]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = 0 if col in {"market_id", "rate"} else ""
    df["market_id"] = _to_numeric(df["market_id"]).fillna(0).astype("int64")
    df["rate"] = _to_numeric(df["rate"]).fillna(0.0)
    return df[columns].sort_values(["symbol", "exchange"]).reset_index(drop=True)


def normalize_recent_trades(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = list(_records(payload, "trades"))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp_ms"] = df["timestamp"].map(to_epoch_ms).astype("int64")
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert(None)
    for col in ["price", "size", "usd_amount", "ask_account_pnl", "bid_account_pnl"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col]).fillna(0.0)
    return df.sort_values("timestamp_ms" if "timestamp_ms" in df.columns else df.columns[0]).reset_index(drop=True)


def normalize_order_book_orders(payload: Mapping[str, Any]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for side in ["asks", "bids"]:
        rows = list(_records(payload, side))
        if not rows:
            continue
        side_df = pd.DataFrame(rows)
        side_df.insert(0, "side", side[:-1])
        frames.append(side_df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in ["initial_base_amount", "remaining_base_amount", "price"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col]).fillna(0.0)
    return df


def candles_to_model_ohlcv(candles: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Convert Lighter candles into the OHLCV CSV shape used by DataPreprocessor."""

    if resolution not in RESOLUTION_TO_MS:
        raise ValueError(f"Unsupported Lighter candle resolution: {resolution}")

    interval_ms = RESOLUTION_TO_MS[resolution]
    out = pd.DataFrame(
        {
            "open_time": candles["timestamp_ms"].astype("int64"),
            "open": candles["open"].astype(float),
            "high": candles["high"].astype(float),
            "low": candles["low"].astype(float),
            "close": candles["close"].astype(float),
            "volume": candles["volume"].astype(float),
            "close_time": candles["timestamp_ms"].astype("int64") + interval_ms - 1,
            "quote_asset_volume": candles["quote_asset_volume"].astype(float),
            "number_of_trades": 0,
            "taker_buy_base_asset_volume": 0.0,
            "taker_buy_quote_asset_volume": 0.0,
            "ignore": 0,
        }
    )
    return out


def write_model_ohlcv_csv(candles: pd.DataFrame, path: Path, resolution: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    candles_to_model_ohlcv(candles, resolution).to_csv(path, header=False, index=False)
    return path


class LighterDataClient:
    """Small REST client for public Lighter market data endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        request_pause_seconds: float = 0.1,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_pause_seconds = request_pause_seconds
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "ethpredict-lighter-data/0.1"})

    def _get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=_drop_none(params or {}), timeout=self.timeout)

        try:
            payload = response.json()
        except ValueError as exc:
            raise LighterApiError(f"Lighter returned non-JSON response from {path}: {response.text[:200]}") from exc

        if response.status_code != 200:
            message = payload.get("message", response.text[:200]) if isinstance(payload, dict) else response.text[:200]
            raise LighterApiError(f"Lighter HTTP {response.status_code} for {path}: {message}")

        code = payload.get("code") if isinstance(payload, dict) else None
        if code not in (None, 200):
            raise LighterApiError(f"Lighter API code {code} for {path}: {payload.get('message', '')}")

        return payload

    def order_book_details(self, market_id: Optional[int] = None, market_filter: str = "all") -> Dict[str, Any]:
        return self._get("/api/v1/orderBookDetails", {"market_id": market_id, "filter": market_filter})

    def resolve_market_id(self, symbol: str = "ETH", market_type: str = "perp") -> int:
        payload = self.order_book_details(market_filter=market_type)
        symbol_upper = symbol.upper()
        key = "spot_order_book_details" if market_type == "spot" else "order_book_details"
        candidates = list(_records(payload, key))

        for market in candidates:
            if str(market.get("symbol", "")).upper() == symbol_upper:
                return int(market["market_id"])

        known = sorted(str(market.get("symbol")) for market in candidates[:20])
        raise LighterApiError(f"Could not find Lighter {market_type} market for {symbol}. Sample symbols: {known}")

    def candles(
        self,
        market_id: int,
        resolution: str,
        start_timestamp: int,
        end_timestamp: int,
        count_back: int,
        set_timestamp_to_end: bool = False,
    ) -> pd.DataFrame:
        payload = self._get(
            "/api/v1/candles",
            {
                "market_id": market_id,
                "resolution": resolution,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "count_back": count_back,
                "set_timestamp_to_end": str(set_timestamp_to_end).lower(),
            },
        )
        return normalize_candles(payload)

    def mark_price_candles(
        self,
        market_id: int,
        resolution: str,
        start_timestamp: int,
        end_timestamp: int,
        count_back: int,
    ) -> pd.DataFrame:
        payload = self._get(
            "/api/v1/markPriceCandles",
            {
                "market_id": market_id,
                "resolution": resolution,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "count_back": count_back,
            },
        )
        return normalize_candles(payload)

    def fundings(
        self,
        market_id: int,
        resolution: str,
        start_timestamp: int,
        end_timestamp: int,
        count_back: int,
    ) -> pd.DataFrame:
        if resolution not in FUNDING_RESOLUTIONS:
            raise ValueError("Lighter fundings only support 1h and 1d resolutions")
        payload = self._get(
            "/api/v1/fundings",
            {
                "market_id": market_id,
                "resolution": resolution,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "count_back": count_back,
            },
        )
        return normalize_fundings(payload)

    def funding_rates(self) -> pd.DataFrame:
        return normalize_funding_rates(self._get("/api/v1/funding-rates"))

    def order_book_orders(self, market_id: int, limit: int = 100) -> pd.DataFrame:
        payload = self._get("/api/v1/orderBookOrders", {"market_id": market_id, "limit": limit})
        return normalize_order_book_orders(payload)

    def recent_trades(self, market_id: int, limit: int = 100) -> pd.DataFrame:
        payload = self._get("/api/v1/recentTrades", {"market_id": market_id, "limit": limit})
        return normalize_recent_trades(payload)

    def exchange_metrics(
        self,
        period: str,
        kind: str,
        market_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        payload = self._get(
            "/api/v1/exchangeMetrics",
            {
                "period": period,
                "kind": kind,
                "filter": "byMarket" if market_symbol else None,
                "value": market_symbol,
            },
        )
        rows = list(_records(payload, "metrics"))
        if not rows:
            return pd.DataFrame(columns=["timestamp_ms", "timestamp", "data"])
        df = pd.DataFrame(rows)
        df["timestamp_ms"] = df["timestamp"].map(to_epoch_ms).astype("int64")
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert(None)
        df["data"] = _to_numeric(df["data"]).fillna(0.0)
        return df[["timestamp_ms", "timestamp", "data"]].sort_values("timestamp_ms").reset_index(drop=True)

    def fetch_candles_range(
        self,
        market_id: int,
        resolution: str,
        start_timestamp: int,
        end_timestamp: int,
        count_back: int = MAX_CANDLES_PER_CALL,
        set_timestamp_to_end: bool = False,
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for window in build_candle_windows(start_timestamp, end_timestamp, resolution, count_back):
            frame = self.candles(
                market_id=market_id,
                resolution=resolution,
                start_timestamp=window.start_timestamp,
                end_timestamp=window.end_timestamp,
                count_back=window.count_back,
                set_timestamp_to_end=set_timestamp_to_end,
            )
            if not frame.empty:
                frame = frame[
                    (frame["timestamp_ms"] >= window.start_timestamp)
                    & (frame["timestamp_ms"] < window.end_timestamp)
                ]
                frames.append(frame)
            if self.request_pause_seconds:
                time.sleep(self.request_pause_seconds)

        return _concat_time_frames(frames)

    def fetch_mark_price_candles_range(
        self,
        market_id: int,
        resolution: str,
        start_timestamp: int,
        end_timestamp: int,
        count_back: int = MAX_CANDLES_PER_CALL,
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for window in build_candle_windows(start_timestamp, end_timestamp, resolution, count_back):
            frame = self.mark_price_candles(
                market_id=market_id,
                resolution=resolution,
                start_timestamp=window.start_timestamp,
                end_timestamp=window.end_timestamp,
                count_back=window.count_back,
            )
            if not frame.empty:
                frame = frame[
                    (frame["timestamp_ms"] >= window.start_timestamp)
                    & (frame["timestamp_ms"] < window.end_timestamp)
                ]
                frames.append(frame)
            if self.request_pause_seconds:
                time.sleep(self.request_pause_seconds)

        return _concat_time_frames(frames)

    def fetch_fundings_range(
        self,
        market_id: int,
        resolution: str,
        start_timestamp: int,
        end_timestamp: int,
        count_back: int = MAX_CANDLES_PER_CALL,
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for window in build_candle_windows(start_timestamp, end_timestamp, resolution, count_back):
            frame = self.fundings(
                market_id=market_id,
                resolution=resolution,
                start_timestamp=window.start_timestamp,
                end_timestamp=window.end_timestamp,
                count_back=window.count_back,
            )
            if not frame.empty:
                frame = frame[
                    (frame["timestamp_ms"] >= window.start_timestamp)
                    & (frame["timestamp_ms"] < window.end_timestamp)
                ]
                frames.append(frame)
            if self.request_pause_seconds:
                time.sleep(self.request_pause_seconds)

        return _concat_time_frames(frames)


def _concat_time_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame()
    merged = pd.concat(usable, ignore_index=True)
    return merged.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
