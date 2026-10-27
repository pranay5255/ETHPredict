# Funding-cost ground truth: issue #54 — 2026-10-27

The funding-cost grader must be frozen on complete, identified source history.
Freezing rewards with 31 known holes per market would require recomputing them
after recovery. This fix repairs collection and the existing lake before another
research freeze, and rejects missing scheduled payments when constructing the
dynamic cost model.

## Root cause and implementation

Six public ETH probes reproduced the boundary failure. A three-hour request with
three `count_back` slots returned two rows; a 500-hour request with 500 slots
returned 499. With unchanged time bounds and one extra slot, the responses held
all three/all 500 rows, including the previously dropped start. Wider overlapping
requests also returned the boundary rows. Request parameters, counts and response
hashes are in [the recovery audit](funding_gap_recovery_20261027.json); complete
responses are preserved in the ignored recovery directory.

`fetch_fundings_range` now overlaps the previous interval and reserves two slots
for the overlap/endpoint. Requests remain at most 502 records, below the
[documented funding limit of 750](https://apidocs.lighter.xyz/reference/fundings).
Returned rows are trimmed to the requested half-open window and deduplicated.
Funding normalization rejects invalid rates/directions and conflicting duplicates
instead of converting missing values to zero.

`lighter_backfill_lake.py --recover-funding-gaps` inspects existing observed
bounds, queries only missing hourly payments, saves source receipts, backs up the
original bytes, adds recovered records and refreshes funding manifests. It
preserves existing rows and unrelated dataset identities. Empty responses remain
unresolved and fail the CLI; source failure aborts before any partition changes
for that market. Completed recovery is idempotent.

`CostModel` checks every declared payment period, including the initial causal
quote and final scheduled payment. A relaxed `max_gap` cannot authorize a missing
scheduled funding payment. Explicit zero source rates remain valid.

## Actual recovery and independent checks

- ETH/BTC/SOL/HYPE/NEAR: **31 gaps each → zero**. All **155** rows came from the
  public endpoint; no unresolved timestamp or invented zero remains.
- All **25 dataset directories** load directly with `pd.read_parquet(directory)`;
  all **425 partition hashes** match current manifests. There are no duplicate
  timestamps. Bounds remain those of the original snapshot through October 16.
- **85 funding partitions** changed; all **340 nonfunding partitions** retained
  their hashes. Original manifests and changed partition bytes are backed up and
  verified against the pre-repair hashes. Every original payment is unchanged.
- All six older ETH archive records agree with recovered rates and directions.
- **432** historical raw/side-data and bounded/development experiment files retain
  their hashes, including models, predictions, results and the final-test ledger.
  No training, reward rerun or final-test evaluation was performed.

Local receipts/backups: `data/lake/lighter/_funding_recovery/20261027T170239414391Z/`.
The lake and full receipts remain ignored; the compact committed audit contains
recovered rows, request evidence, partition identities and protection hashes.
The original `lighter_lake_audit_20261017.json` is preserved as dated evidence.

## Verification

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q`:
**317 passed, 23 warnings in 116.44s**, including project GPU smoke tests and the
available vendored SDK tests. Warnings are two SDK websocket deprecations and
constant-series statistics. New fixtures cover month starts, 500-hour boundaries,
inclusive/exclusive responses, duplicates, empty responses, closed-month recovery,
idempotence, preservation/tampering, source failure, invalid rates and grading
rejection of missing initial/internal/final payments. `git diff --check` passed.

Issue #54 can close when this fix is merged. General lake ingestion/export and
new economic experiments remain separate work.
