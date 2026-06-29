"""Collect public Lighter market data for ETHPredict training.

Examples:
    python scripts/lighter_collect_data.py --start 2025-01-17 --end 2025-01-24
    python scripts/lighter_collect_data.py --config configs/config.yml --resolution 5m
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.lighter_client import (
    DEFAULT_BASE_URL,
    MAX_CANDLES_PER_CALL,
    FUNDING_RESOLUTIONS,
    LighterDataClient,
    to_epoch_ms,
    write_model_ohlcv_csv,
)


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _lighter_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("data", {}).get("lighter", {}) or {}


def _default_end_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _date_label(value_ms: int) -> str:
    return pd.to_datetime(value_ms, unit="ms", utc=True).strftime("%Y%m%d")


def _write_frame(df: pd.DataFrame, path: Path) -> Optional[Path]:
    if df.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect public Lighter data for model training")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yml"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--symbol", default=None, help="Lighter market symbol, e.g. ETH")
    parser.add_argument("--market-type", default=None, choices=["perp", "spot"])
    parser.add_argument("--market-id", type=int, default=None)
    parser.add_argument("--pipeline-symbol", default=None, help="Raw CSV prefix expected by DataPreprocessor")
    parser.add_argument("--resolution", default=None, choices=["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"])
    parser.add_argument("--funding-resolution", default=None, choices=["1h", "1d"])
    parser.add_argument("--start", default=None, help="UTC date/datetime, seconds, or milliseconds")
    parser.add_argument("--end", default=None, help="UTC date/datetime, seconds, or milliseconds")
    parser.add_argument("--count-back", type=int, default=None)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--side-dir", type=Path, default=Path("data/lighter"))
    parser.add_argument("--request-pause-seconds", type=float, default=None)
    parser.add_argument("--skip-side-data", action="store_true")
    parser.add_argument("--skip-mark-price", action="store_true")
    parser.add_argument("--order-book-limit", type=int, default=100)
    parser.add_argument("--recent-trades-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    lighter_cfg = _lighter_config(config)

    base_url = args.base_url or lighter_cfg.get("base_url", DEFAULT_BASE_URL)
    symbol = args.symbol or lighter_cfg.get("symbol", "ETH")
    market_type = args.market_type or lighter_cfg.get("market_type", "perp")
    pipeline_symbol = args.pipeline_symbol or lighter_cfg.get("pipeline_symbol", "ETHUSDT")
    resolution = args.resolution or lighter_cfg.get("resolution", "1h")
    funding_resolution = args.funding_resolution or lighter_cfg.get("funding_resolution", "1h")
    count_back = args.count_back or lighter_cfg.get("count_back", MAX_CANDLES_PER_CALL)
    start = args.start or lighter_cfg.get("start_date", "2025-01-17")
    end = args.end or lighter_cfg.get("end_date", _default_end_date())
    request_pause_seconds = args.request_pause_seconds
    if request_pause_seconds is None:
        request_pause_seconds = lighter_cfg.get("request_pause_seconds", 0.1)

    start_ms = to_epoch_ms(start)
    end_ms = to_epoch_ms(end)
    client = LighterDataClient(base_url=base_url, request_pause_seconds=request_pause_seconds)
    market_id = args.market_id if args.market_id is not None else lighter_cfg.get("market_id")
    if market_id is None:
        market_id = client.resolve_market_id(symbol=symbol, market_type=market_type)
    market_id = int(market_id)

    print(
        f"Collecting Lighter {symbol} {market_type} market_id={market_id} "
        f"{resolution} candles from {start} to {end}"
    )
    candles = client.fetch_candles_range(
        market_id=market_id,
        resolution=resolution,
        start_timestamp=start_ms,
        end_timestamp=end_ms,
        count_back=count_back,
    )
    if candles.empty:
        raise SystemExit("No Lighter candles returned for the requested range")

    start_label = _date_label(start_ms)
    end_label = _date_label(end_ms)
    raw_path = args.raw_dir / f"{pipeline_symbol}-{resolution}-lighter-{start_label}-{end_label}.csv"
    native_candles_path = args.side_dir / f"candles_{symbol}_{market_type}_{resolution}_{start_label}_{end_label}.csv"

    write_model_ohlcv_csv(candles, raw_path, resolution)
    _write_frame(candles, native_candles_path)
    print(f"Wrote model OHLCV CSV: {raw_path}")
    print(f"Wrote native candles: {native_candles_path}")

    if args.skip_side_data:
        return

    if not args.skip_mark_price:
        mark_prices = client.fetch_mark_price_candles_range(
            market_id=market_id,
            resolution=resolution,
            start_timestamp=start_ms,
            end_timestamp=end_ms,
            count_back=count_back,
        )
        path = _write_frame(
            mark_prices,
            args.side_dir / f"mark_price_candles_{symbol}_{market_type}_{resolution}_{start_label}_{end_label}.csv",
        )
        if path:
            print(f"Wrote mark price candles: {path}")

    if funding_resolution in FUNDING_RESOLUTIONS:
        fundings = client.fetch_fundings_range(
            market_id=market_id,
            resolution=funding_resolution,
            start_timestamp=start_ms,
            end_timestamp=end_ms,
            count_back=count_back,
        )
        path = _write_frame(
            fundings,
            args.side_dir / f"fundings_{symbol}_{market_type}_{funding_resolution}_{start_label}_{end_label}.csv",
        )
        if path:
            print(f"Wrote fundings: {path}")

    funding_rates = client.funding_rates()
    path = _write_frame(funding_rates, args.side_dir / "funding_rates_latest.csv")
    if path:
        print(f"Wrote latest funding rates: {path}")

    details_payload = client.order_book_details(market_id=market_id, market_filter=market_type)
    details_path = args.side_dir / f"order_book_details_{symbol}_{market_type}.json"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(details_payload, indent=2))
    print(f"Wrote order book details: {details_path}")

    order_book = client.order_book_orders(market_id=market_id, limit=args.order_book_limit)
    path = _write_frame(order_book, args.side_dir / f"order_book_orders_{symbol}_{market_type}_latest.csv")
    if path:
        print(f"Wrote order book snapshot: {path}")

    recent_trades = client.recent_trades(market_id=market_id, limit=args.recent_trades_limit)
    path = _write_frame(recent_trades, args.side_dir / f"recent_trades_{symbol}_{market_type}_latest.csv")
    if path:
        print(f"Wrote recent trades snapshot: {path}")

    optional_metrics = [
        ("volume", f"exchange_metric_volume_{symbol}_{market_type}_all.csv"),
        ("open_interest", f"exchange_metric_open_interest_{symbol}_{market_type}_all.csv"),
    ]
    for metric_kind, filename in optional_metrics:
        try:
            metric_frame = client.exchange_metrics(period="all", kind=metric_kind, market_symbol=symbol)
        except Exception as exc:
            print(f"Skipping optional exchange metric {metric_kind}: {exc}")
            continue
        path = _write_frame(metric_frame, args.side_dir / filename)
        if path:
            print(f"Wrote exchange metric {metric_kind}: {path}")


if __name__ == "__main__":
    main()
