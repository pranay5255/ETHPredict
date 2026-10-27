"""Backfill public Lighter perp history into a git-ignored parquet lake.

Usage:
    python scripts/lighter_backfill_lake.py --symbols ETH BTC SOL HYPE NEAR
    python scripts/lighter_backfill_lake.py --env-file .env.testnet --symbols ETH BTC SOL
    python scripts/lighter_backfill_lake.py --repair --symbols ETH BTC   # rewrite schema, no network

``--env-file`` reads only ``LIGHTER_BASE_URL`` and ``LIGHTER_NETWORK``; API keys in
the file are ignored. Data from any network other than mainnet goes to
``data/lake/lighter-<network>/`` so test data never mixes with real market history.

Layout: ``data/lake/lighter/<SYMBOL>/<dataset>/<YYYY-MM>.parquet`` where dataset is
``candles_1m``, ``candles_5m``, ``candles_1h``, ``mark_price_5m``, or ``fundings_1h``.
Completed months are skipped on rerun and the current month is refreshed, so an
interrupted backfill resumes where it stopped. Each market gets ``_manifest.json``
with row counts, interval gaps, and partition hashes. Public read-only endpoints
only; no keys or orders. Requests are paced; if Lighter answers with its
human-verification page the script waits, then stops so a later rerun resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import dotenv_values
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.lighter_client import DEFAULT_BASE_URL, RESOLUTION_TO_MS, LighterApiError, LighterDataClient  # noqa: E402

DATASETS = {
    "candles_1m": ("candles", "1m"),
    "candles_5m": ("candles", "5m"),
    "candles_1h": ("candles", "1h"),
    "mark_price_5m": ("mark_price", "5m"),
    "fundings_1h": ("fundings", "1h"),
}


def retrying_session() -> requests.Session:
    retry = Retry(total=8, connect=8, read=8, status=8, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
                  respect_retry_after_header=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class PoliteLighterClient(LighterDataClient):
    """Back off when the API serves its verification page instead of JSON."""

    challenge_waits = (60, 180, 600)

    def _get(self, path, params=None):
        for wait in (*self.challenge_waits, None):
            try:
                return super()._get(path, params)
            except (LighterApiError, requests.exceptions.JSONDecodeError) as exc:
                if "non-JSON" not in str(exc) and not isinstance(exc, requests.exceptions.JSONDecodeError):
                    raise
                if wait is None:
                    raise SystemExit("Lighter keeps serving its verification page; rerun later to resume") from exc
                print(f"  verification page from {path}; waiting {wait}s", flush=True)
                time.sleep(wait)


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    return list(pd.date_range(start.to_period("M").to_timestamp(), end, freq="MS"))


def to_ms(value: pd.Timestamp) -> int:
    return int(value.tz_localize("UTC").timestamp() * 1000) if value.tzinfo is None else int(value.timestamp() * 1000)


def fetch(client: LighterDataClient, kind: str, market_id: int, resolution: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    if kind == "candles":
        return client.fetch_candles_range(market_id, resolution, start_ms, end_ms)
    if kind == "mark_price":
        return client.fetch_mark_price_candles_range(market_id=market_id, resolution=resolution,
                                                     start_timestamp=start_ms, end_timestamp=end_ms)
    return client.fetch_fundings_range(market_id=market_id, resolution=resolution,
                                       start_timestamp=start_ms, end_timestamp=end_ms)


def first_listed_day(client: LighterDataClient, market_id: int) -> pd.Timestamp:
    now_ms = int(time.time() * 1000)
    daily = client.candles(market_id=market_id, resolution="1d", start_timestamp=now_ms - 1_000 * 86_400_000,
                           end_timestamp=now_ms, count_back=500)
    if daily.empty:
        raise SystemExit(f"No daily candles for market {market_id}")
    return pd.Timestamp(int(daily["timestamp_ms"].min()), unit="ms")


CANDLE_COLUMNS = ["timestamp_ms", "timestamp", "open", "high", "low", "close", "volume", "quote_asset_volume",
                  "last_trade_id", "sample_count", "open_raw", "high_raw", "low_raw", "close_raw", "market_id", "symbol"]
COLUMNS = {"candles": CANDLE_COLUMNS, "mark_price": CANDLE_COLUMNS,
           "fundings": ["timestamp_ms", "timestamp", "value", "rate", "direction", "market_id", "symbol"]}
INT_COLUMNS = {"timestamp_ms", "last_trade_id", "sample_count", "market_id"}
TEXT_COLUMNS = {"symbol", "direction"}


def conform(frame: pd.DataFrame, kind: str, market_id: int, symbol: str) -> pd.DataFrame:
    """One schema for every partition, so a dataset directory reads as one table.

    Lighter omits zero-valued fields, so a column absent for a whole month would
    otherwise be saved as integers while other months hold floats.
    """
    out = frame.copy()
    out["market_id"], out["symbol"] = market_id, symbol
    for column in COLUMNS[kind]:
        if column not in out:
            out[column] = pd.Series(index=out.index, dtype="object")
    for column in COLUMNS[kind]:
        if column in INT_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype("int64")
        elif column in TEXT_COLUMNS:
            out[column] = out[column].fillna("").astype(str)
        elif column != "timestamp":
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    out["timestamp"] = pd.to_datetime(out["timestamp_ms"], unit="ms")
    return out[COLUMNS[kind]]


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def summarize(dataset_dir: Path, resolution: str) -> Dict[str, object]:
    parts = sorted(dataset_dir.glob("*.parquet"))
    frames = {part.name: pd.read_parquet(part) for part in parts}
    stamps = pd.concat([frame["timestamp_ms"] for frame in frames.values() if "timestamp_ms" in frame and len(frame)],
                       ignore_index=True) if frames else pd.Series(dtype="int64")
    empty = sorted(name for name, frame in frames.items() if "timestamp_ms" not in frame or not len(frame))
    if not len(stamps):
        return {"rows": 0, "empty_partitions": empty}
    stamps = stamps.sort_values()
    interval = RESOLUTION_TO_MS[resolution]
    steps = stamps.diff().dropna()
    return {
        "rows": int(len(stamps)),
        "empty_partitions": empty,
        "duplicates": int(stamps.duplicated().sum()),
        "first": str(pd.Timestamp(int(stamps.iloc[0]), unit="ms")),
        "last": str(pd.Timestamp(int(stamps.iloc[-1]), unit="ms")),
        "expected_interval_ms": interval,
        "missing_intervals": int(((steps // interval) - 1).clip(lower=0).sum()),
        "largest_gap_intervals": int(steps.max() // interval) if len(steps) else 0,
        "partitions": {part.name: hashlib.sha256(part.read_bytes()).hexdigest() for part in parts},
    }


def backfill_market(client: LighterDataClient, symbol: str, root: Path, datasets: List[str], start_override: str | None,
                    network: str = "mainnet") -> None:
    market_id = client.resolve_market_id(symbol, "perp")
    listed = first_listed_day(client, market_id)
    start = max(listed, pd.Timestamp(start_override)) if start_override else listed
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    current_month = now.to_period("M").to_timestamp()
    market_dir = root / symbol
    print(f"[{symbol}] market_id={market_id} listed={listed:%Y-%m-%d} backfill from {start:%Y-%m-%d %H:%M}", flush=True)
    for name in datasets:
        kind, resolution = DATASETS[name]
        interval = RESOLUTION_TO_MS[resolution]
        for month in month_starts(start, now):
            path = market_dir / name / f"{month:%Y-%m}.parquet"
            if path.exists() and month < current_month:
                continue
            month_end = month + pd.offsets.MonthBegin(1)
            begin_ms, end_ms = to_ms(max(month, start)), to_ms(min(month_end, now))
            end_ms -= end_ms % interval  # completed intervals only
            if end_ms <= begin_ms:
                continue
            started = time.time()
            frame = fetch(client, kind, market_id, resolution, begin_ms, end_ms)
            frame = frame[(frame["timestamp_ms"] >= begin_ms) & (frame["timestamp_ms"] < end_ms)] if len(frame) else frame
            write_atomic(conform(frame, kind, market_id, symbol), path)
            print(f"[{symbol}] {name} {month:%Y-%m}: {len(frame):6d} rows ({time.time() - started:4.0f}s)", flush=True)
    manifest = {
        "symbol": symbol,
        "market_id": market_id,
        "market_type": "perp",
        "network": network,
        "source": client.base_url,
        "listed": str(listed),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {name: summarize(market_dir / name, DATASETS[name][1]) for name in datasets},
    }
    (market_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[{symbol}] manifest written", flush=True)


def repair_market(root: Path, symbol: str) -> None:
    """Rewrite existing partitions into the shared schema; no network access."""
    market_dir = root / symbol
    manifest_path = market_dir / "_manifest.json"
    if not manifest_path.exists():
        print(f"[{symbol}] no manifest in {market_dir}; skipped (backfill missing or still running)", flush=True)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, (kind, resolution) in DATASETS.items():
        parts = sorted((market_dir / name).glob("*.parquet"))
        for part in parts:
            write_atomic(conform(pd.read_parquet(part), kind, int(manifest["market_id"]), symbol), part)
        if parts:
            manifest["datasets"][name] = summarize(market_dir / name, resolution)
    manifest["repaired_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[{symbol}] partitions conformed and manifest refreshed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", nargs="+", default=["ETH"])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--env-file", type=Path, default=None,
                        help="Read LIGHTER_BASE_URL and LIGHTER_NETWORK from this file (e.g. .env.testnet); keys are ignored")
    parser.add_argument("--base-url", default=None, help="Overrides --env-file; defaults to Lighter mainnet")
    parser.add_argument("--root", type=Path, default=None,
                        help="Defaults to data/lake/lighter on mainnet, data/lake/lighter-<network> otherwise")
    parser.add_argument("--start", default=None, help="Optional UTC start; defaults to each market's listing day")
    parser.add_argument("--request-pause-seconds", type=float, default=0.5)
    parser.add_argument("--repair", action="store_true",
                        help="Only rewrite existing partitions into the shared schema and refresh manifests (no network)")
    args = parser.parse_args()

    settings = dotenv_values(args.env_file) if args.env_file else {}
    base_url = (args.base_url or settings.get("LIGHTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    network = settings.get("LIGHTER_NETWORK") or next(
        (name for name in ("mainnet", "testnet") if name in base_url), "custom")
    mainnet_root = Path("data/lake/lighter")
    root = args.root or (mainnet_root if network == "mainnet" else Path(f"data/lake/lighter-{network}"))
    if network != "mainnet" and root.resolve() == mainnet_root.resolve():
        raise SystemExit(f"Refusing to write {network} data into the mainnet lake {mainnet_root}")
    if args.repair:
        for symbol in args.symbols:
            repair_market(root, symbol.upper())
        return
    print(f"Source: {base_url} ({network}) -> {root}", flush=True)
    if network != "mainnet":
        print("Note: non-mainnet candles reflect test activity and short history; train models on mainnet data.", flush=True)

    client = PoliteLighterClient(base_url=base_url, request_pause_seconds=args.request_pause_seconds,
                                 session=retrying_session())
    for symbol in args.symbols:
        backfill_market(client, symbol.upper(), root, args.datasets, args.start, network)


if __name__ == "__main__":
    main()
