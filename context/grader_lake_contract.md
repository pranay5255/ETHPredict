# P1 lake contract and verification

P0 merged in #60. The standalone `grader.data` implementation imports no editable
`src`/`strategy` modules. It does not alter the active CSV training path, historical
inputs, manifests or cost/statistics implementations.

## Contract

`LakeLoader(root).load(market, dataset, resolution, start, end)` returns a
`LoadedDataset(frame, provenance)`. `load_markets([...], ...)` returns
`MarketData`, preserving each market's frame/provenance and a combined `frame`.
Datasets/resolutions are explicit (`candles_1m/1m`, `candles_5m/5m`,
`candles_1h/1h`, `fundings_1h/1h`, `mark_price_5m/5m`). Dates and offsets
normalize to UTC; windows are half-open on integer source `timestamp_ms`.
Invalid/ambiguous numeric dates and empty windows are errors.

The loader reads only intersecting monthly partitions, verifies SHA-256 before
Parquet deserialization and deserializes those exact bytes to avoid a hash/read
race. Manifest names and resolved source paths cannot escape the lake root.
Missing files/hashes, conflicting manifest keys, invalid schemas, off-grid
timestamps, identity mismatches and nonfinite source values are errors. Same-time
rows must agree in every source column; only exact duplicates collapse.

Provenance includes market/ID, dataset/resolution, requested UTC window, manifest
hash, partition files/hashes/physical and selected counts/bounds, unique selected
row count, first/last times, and compressed missing runs including edge gaps.
No source record is synthesized. Required gaps reject by default.
`require_complete=False` is an explicit diagnostic mode, not permission for a
future scorer to accept a gap; it preserves missing runs and missing rows.

`available_at_ms` is bar-open plus resolution for candles/mark and the payment
timestamp for funding. `availability_join(decision_times, loaded, columns)` uses
only available records and expires them at the next expected publication. It
does not carry a stale record across a missing bar/payment. Unmatched values
are NaN with `available=False`, never zero. Callers load prior context explicitly.
Mark-price history before its **actual** first observation (August 25, 2025,
15:25 UTC) remains unavailable; the loader allows this known leading absence
but rejects unexpected internal/final gaps by default.

Prices are quote currency per base asset, `volume` is base asset volume, and
`quote_asset_volume` is quote asset volume. Neither CSV format needs scaling.
The headerless model CSV's `open_time` equals `timestamp_ms` and its inclusive
`close_time` equals `timestamp_ms + resolution - 1ms`; the availability clock is
the following millisecond. Funding `rate` is the existing source magnitude with
explicit long/short payer direction, not an implicitly signed decimal fraction;
the existing CostModel's explicit unit/direction settings still govern cashflows.

## Acceptance evidence

```text
ETHPREDICT_SOURCE_REPO=<original checkout> OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <existing .venv>/bin/python \
-m pytest tests/test_grader_data.py -q
38 passed in 0.99s

<existing .venv>/bin/python -m scripts.verify_lake_csv_parity \
--repo <original checkout> --output context/grader_lake_csv_parity.json
rows: 26496; different_cells: 0
```

The verifier compares the **entire** March 28–June 28, 2026 common window,
matching both the normalized Lighter archive and the actual headerless model
export to the lake, exactly in float64 for OHLC, base volume and quote volume.
`float_precision='round_trip'` reconstructs the numbers represented by CSV text;
there are no fitted conversions or tolerances hiding source disagreement.
It checks all timestamps and the inclusive model close clock. The committed
receipt contains CSV and partition hashes, row counts and window bounds.
It does not run training or consume a test.

The same integration test loads all five markets for March 28, 2026 and verifies
288 rows each and complete independent provenance. The ordinary fixture suite
covers pre-deserialization hashing, hash/read races, missing files/hashes,
duplicates/conflicts, invalid/missing/unknown values/columns, half-open/timezone
windows, source availability, funding payment timing, stale-data expiration,
mark-price leading absence and symlink traversal. The local-lake integration
test requires `ETHPREDICT_SOURCE_REPO`; synthetic contract tests run without it.

With the existing environment and the pinned SDK submodule initialized, the
full suite also passes (the local integration check was enabled):

```text
UV_PROJECT_ENVIRONMENT=<existing .venv> ETHPREDICT_SOURCE_REPO=<original checkout> \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync pytest -q
355 passed, 23 warnings in 115.43s (0:01:55)
git diff --check: passed
```

No dependencies were added or upgraded. The 23 warnings are existing SDK
websocket deprecations and constant-series statistics.

## New source integrity finding

A broader dev-only scan of all five datasets in all five markets validates
23 of 25 market/dataset contracts. HYPE's October 2025 `candles_1m` has four rows
with invalid OHLC geometry, and `candles_5m` has one. Example: the October 27
20:50 UTC 5m bar has open 48.659 but high 48.646. These are source records, not a
CSV/lake parity disagreement. `context/grader_dev_contract_audit.json` records
the 23 successes and two fail-closed rejections. No source bytes are repaired,
values clipped, markets silently dropped or gates relaxed. A strategy requiring
those malformed partitions will fail data integrity until an independently
reviewed data repair is authorized. ETH CSV/lake parity passes unchanged.

## Approved clarifications

The human approved on 2026-10-27: calibrate baseline **pre-gate components** to
the requested range while preserving every gate and rejected final score 0;
and allow an isolated private strategy worker to consume causal hidden-tier
slices, with all hidden inputs/logs/positions/artifacts inaccessible to the
loop agent. Those are design decisions, not completed isolation tests.

P2 physical sealing, P3 runtime/scoring/baselines/freeze, P4 loop and P5 seeds
remain outstanding. No strategy edge or secure hidden evaluation is claimed.
