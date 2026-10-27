# Lighter data setup — 2026-10-17

Public data collection uses REST without account credentials or signed orders.
The primary research market is ETH perpetual (`market_id: 0`). Collection now
also covers BTC, SOL, HYPE and NEAR in a separate historical Parquet lake.

## Existing CSV research path

Use the uv-managed environment and date-bounded collector:

```bash
uv sync
uv run python scripts/lighter_collect_data.py --config configs/config.yml
```

It writes model OHLCV to
`data/raw/ETHUSDT-<resolution>-lighter-<start>-<end>.csv` and side tables/snapshots
under `data/lighter/`. `DataPreprocessor` deliberately accepts the Lighter raw
filename pattern. Build features from these CSVs:

```bash
uv run python -m src.data.features_all full_features --data-dir data --granularity 5m --sequence-length 288 --out-dir /tmp/ethpredict-lighter-features
```

Historical side groups are config-controlled in `data.side_data.groups` and use
availability-time alignment plus per-split coverage gates. None is enabled in the
default config. Latest order-book/recent-trade snapshots are not historical tape.

## Mainnet history lake

```bash
uv run python scripts/lighter_backfill_lake.py --symbols ETH BTC SOL HYPE NEAR
```

The completed snapshot spans June 4, 2025 through a partial October 16, 2026:
1m/5m/1h candles, 5m mark candles (from August 25, 2025) and hourly funding.
All 25 dataset directories load with `pd.read_parquet(directory)` and all 425
partition hashes match manifests. Pre-recovery size: 303.10 MB (289.06 MiB).

The [#54](https://github.com/pranay5255/ETHPredict/issues/54) fix recovered all
31 missing funding hours per market from source (155 total), with original-byte
backups and source receipts. Zero internal gaps remain; code awaits merge. See
[verification](funding_gap_fix_20261027.md). Unknown rates are never zero-filled;
schema repair does not recover missing rows.

See [the lake runbook](lighter_data_lake.md) for exact observed bounds, audit,
repair and rerun semantics. Normal reruns skip existing closed-month files and
refresh the current UTC month. No general training lake reader exists yet; freeze
an explicit provenance-preserving reader/export before using this new history.

## Endpoint roles

`orderBookDetails` supplies market metadata; `candles` supplies OHLCV;
`markPriceCandles` and `fundings` supply historical side data. `funding-rates`,
`orderBookOrders` and `recentTrades` supply latest snapshots. Exchange metrics are
best effort and subject to their own historical coverage. Pace public requests
and run one backfill at a time. No historical order-book depth is in this lake.

Separate non-mainnet output under `data/lake/lighter-testnet/`. Keep signed
trading setup in [lighter_integration.md](lighter_integration.md) and the root
README; it is separate from this public collection path.
