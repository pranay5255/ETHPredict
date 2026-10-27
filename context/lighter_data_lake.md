# Lighter mainnet history lake — funding recovery 2026-10-27

Historical collection is complete for ETH, BTC, SOL, HYPE and NEAR through
2026-10-16. Collection completeness is distinct from funding continuity and
training integration. The lake remains local and gitignored. On 2026-10-27,
all **155 missing funding payments** (31 per market) were recovered from the
public source. All 25 directories remain readable, with zero duplicate timestamps
and zero internal funding gaps. See
[the recovery audit](funding_gap_recovery_20261027.json) and
[fix verification](funding_gap_fix_20261027.md).

## Original verified coverage — before recovery

An independent read-only audit loaded all **25 dataset directories**, checked
**425 monthly partition SHA-256 hashes** against five manifests, and verified
uniform partition schemas, row counts, timestamps, symbols and market IDs.
Parquet size is **303,103,519 bytes (303.10 MB; 289.06 MiB)**. Exact per-market
bounds, schema and checksums are in
[lighter_lake_audit_20261017.json](lighter_lake_audit_20261017.json).

| Dataset per market | Rows across markets | First UTC | Internal missing intervals |
| --- | --- | --- | --- |
| 1m candles | 719,697–719,775 | 2025-06-04 00:00 | 0 |
| 5m candles | 143,939–143,955 | 2025-06-04 00:00 | 0 |
| 1h candles | 11,994–11,996 | 2025-06-04 00:00 | 0 |
| 5m mark price | 120,138–120,154 | 2025-08-25 15:25 | 0 after first observation |
| 1h funding | 11,962–11,964 | 2025-06-04 01:00 | 31 per market |

Collection endpoints differ by market and resolution: the last observations are
on October 16 between 17:00 and 20:14 UTC. This is a partial final day, not a
promise of synchronized coverage through midnight. Missing counts measure only
the interval between each dataset's first and last observations. June and July
mark-price partitions are intentionally empty; earlier mark history is unavailable
in this snapshot. Every dataset has zero duplicate timestamps.

## Funding boundary fix and targeted recovery

All five markets originally had the same 31 missing hours. **Six of those hours exist in
the older ETH archive** `data/lighter/fundings_ETH_perp_1h_20260328_20260628.csv`:
April 1 00:00, April 21 20:00, May 1 00:00, May 21 20:00, June 1 00:00 and
June 21 20:00, all in 2026 UTC. Thus identical gaps do not establish unavailable
exchange data. The month-start and 500-hour boundary pattern suggests collection
window behavior. Live requests now reproduce the cause: a 500-hour request with
`count_back=500` returns 499 rows and loses its first hour; reserving one extra
slot returns all 500. The fixed client overlaps one interval and reserves both
that overlap and the endpoint slot, then trims to the requested half-open range.

The [#54](https://github.com/pranay5255/ETHPredict/issues/54) fix adds explicit
recovery for closed partitions:

```bash
uv run python scripts/lighter_backfill_lake.py --recover-funding-gaps --symbols ETH BTC SOL HYPE NEAR
```

Recovery requests only missing hours inside each dataset's original observed
bounds. Existing payments and bounds are retained. Original partition/manifest
bytes, source-response receipts and before/after hashes are saved under
`data/lake/lighter/_funding_recovery/<UTC-run-id>/`. Missing responses remain
unresolved, and the command exits unsuccessfully if any gaps remain. Repeating a
completed recovery makes no source requests or partition changes.

The actual recovery added 155 source rows across 85 funding partitions, leaving
all 340 nonfunding partitions unchanged. All six ETH archive counterexamples
agree with the fetched rates and directions. All 432 enrolled historical input
and experiment files, including the final-test ledger, remain byte-identical.
The dated 2026-10-17 audit above remains the original pre-recovery evidence.

Schema `--repair` does not recover rows, and normal refresh skips closed months.
Unknown funding is never zero-filled. Dynamic cost grading now refuses an
internal missing scheduled payment even with a relaxed `max_gap`, as well as
missing payments at the price-path boundaries. Freeze funding-cost rewards only
after this coverage check passes; source/config hashes identify the actual inputs.

## Collect, refresh and read

Run one public, read-only collector at a time from the repository root:

```bash
uv run python scripts/lighter_backfill_lake.py --symbols ETH BTC SOL HYPE NEAR
```

Existing closed-month files are skipped; the current UTC month is refreshed.
Missing closed-month files are collected. Requests are paced at 0.5 seconds by
default, with retries/backoff. The operator reported no verification pages during
the completed mainnet run; the local audit cannot independently establish that.

```python
import pandas as pd
candles = pd.read_parquet("data/lake/lighter/ETH/candles_5m")
funding = pd.read_parquet("data/lake/lighter/BTC/fundings_1h")
```

Layout is `data/lake/lighter/<SYMBOL>/<dataset>/<YYYY-MM>.parquet`, with
`<SYMBOL>/_manifest.json` containing source, market ID, collection time, row/gap
counts and partition hashes. Repair old schemas locally, without network calls:

```bash
uv run python scripts/lighter_backfill_lake.py --repair --symbols BTC SOL HYPE NEAR
```

The operator reported unchanged row counts, price sums and timestamp sums across
that repair. Current directory reads and manifest agreement are verified; no
independent before-repair snapshot was available to re-prove that invariance.
The manifests also record an ETH repair.

`--env-file` reads only `LIGHTER_BASE_URL` and `LIGHTER_NETWORK`, ignoring API keys.
Testnet belongs under `data/lake/lighter-testnet/`; use it for connectivity checks,
not mainnet research. Preserve the endpoint/network distinction when overriding
`--root` or `--base-url`.

## What this enables next

The active training loader still reads `data/raw/ETHUSDT-5m-lighter-*.csv`.
The lake does not automatically change any existing model, feature matrix,
configuration, split, result or test-use ledger. The PR #53 bounded development
study used three exact lake funding records as explicit repairs to its own frozen
CSV input; that is not a general lake loader.

Before a new lake-backed experiment: verify the repaired funding coverage, implement an
explicit market/date/resolution loader or immutable CSV export with partition
hashes, establish common coverage and unit/availability-time contracts, and freeze
one new development protocol. Start with ETH and a narrow hypothesis; additional
markets are replication opportunities, not independent proof of an edge. Market
selection counts toward the research budget. Keep prior consumed tests protected.

Historical data availability does not complete recurring snapshot #43, frozen
inference #44 or dashboard #45. Live order-book history is not supplied by this
lake. No strategy has met the economic promotion gate.
