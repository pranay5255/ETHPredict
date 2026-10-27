# P0 verification — 2026-10-27

Starting point: `main` at `c365a81`, verified with `git log -8 --oneline`.
`gh pr view 51 --json state,mergeCommit` and `gh pr view 53 --json state`
report MERGED; `gh issue list --state closed --limit 100 --json number,state,title`
includes #16/#17/#20. `gh pr list --state open --json number,title,files`
returned `[]` before work. The primary checkout contains an unrelated untracked
`autoresearch-goal.md`; it was not edited or staged. P0 uses its own worktree.

## Fresh baseline suite on main

```text
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q
317 passed, 23 warnings in 114.73s (0:01:54)
```

This includes deterministic funding month-start/500-hour boundary,
inclusive/exclusive response, duplicate, empty-source, recovery idempotence,
protected partition and source-failure fixtures, and dynamic CostModel rejection
of missing first/internal/final scheduled payments. Warnings are existing SDK
websocket deprecations and constant-series statistics.

## Independent structural audit

From the P0 worktree, using the existing environment without changing dependencies:

```text
$ /home/pranay5255/Documents/ethpredict/ETHPredict/.venv/bin/python \
    scripts/audit_autoresearch_start.py \
    --repo /home/pranay5255/Documents/ethpredict/ETHPredict \
    --output context/autoresearch_start_audit.json
dataset_directories_loaded: 25
partition_hashes_verified: 425
parquet_bytes: 303106677
protected_historical_files_verified: 432
recovered_payments: 155
remaining_internal_funding_gaps: 0
torch: 2.11.0+cu128
timesfm: 2.0.1
pandas: 3.0.4
gpu: NVIDIA GeForce RTX 4090
```

The audit calls `pd.read_parquet(dataset_directory)` on every directory, verifies
all manifest partition hashes/sets/counts/bounds/market identities, checks exact
timestamp spacing and uniqueness, validates finite funding rates/directions,
and compares all 432 enrolled historical file hashes to the committed recovery
audit. It only inspects structure; no policy, holdout or consumed-test evaluation
was run. It is a **pre-sealing** audit, not a command the loop should run later.

The consumed-test ledger has count 1 for the original bounded ETH test; its bytes
are included in the protected-file check. Prior outcomes remain rejected as
recorded in `context/first_bounded_backtest_20261016.json` and
`context/development_continuation_20261017.json`.

## Differences and boundaries

No required starting prerequisite was false. Exact current Parquet size is
303,106,677 bytes (about 289 MiB), consistent with the approximate lake size.
Existing TASKS.md still said #54 was pending merge, although its fix is already
on main; P0 corrects that stale status after closing the issue with evidence.
The original lake's final day has partial observations after October 16 midnight;
the design preserves the specified half-open tier endpoint rather than
extending it. The CSV filename uses compact dates `20260328_20260628`; P1 will
use that actual path, rather than the attachment's hyphenated glob.

See `autoresearch_design.md` for pretraining timing, strict gate calibration
conflict, human sealing and the hidden worker boundary. None is represented as
an implemented or passed P2/P3 acceptance criterion.
