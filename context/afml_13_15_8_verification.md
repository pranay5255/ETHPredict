# Implementation verification: #13, #15 and #8

This change implements the next three issues from the implementation-before-
experiments plan. It starts from merged main `0db731e` (PR #48). The primary
checkout's separate EDA work and root-level roadmap artifacts were preserved.

## Scope delivered

- #13: named baseline and profit-take-only label success, symmetric/asymmetric
  geometry, explicit stop-first/profit-first/exclude ambiguity policy, validated
  OHLC paths, unavailable unfinished outcomes, complete candidate evidence,
  classification/calibration and threshold/coverage diagnostics.
- #15: inclusive timestamp-span purging, all five embargo modes, exact walk-
  forward/purged k-fold/grouped scenarios, recorded row removal decisions,
  nested base OOF inputs for held-out meta evaluation, realized meta `t1`
  purge audits, training-population weights, protected final-test boundaries,
  and explicit validation-only verification mode. Main and benchmark paths
  share the split contract and persist their nested meta audits.
- #8: config-selected funding/mark/OI/exchange-volume/book/flow historical
  sources, declared publication-time backward joins, finite tolerance,
  identical/conflicting duplicate handling, global/outer/inner split coverage,
  source hashes, explicit missingness, completed volume/dollar aggregation,
  excluded terminal partial groups and persisted bars. The label price path
  remains independent of selected model feature families.

## Checks

Baseline before changes: 100 Python tests passed.
Final regression suite: **157 passed**, 33 scientific-library warnings, in 94.65 seconds.
`git diff --check` and compilation of the changed Python modules also passed.

The suite includes GPU environment/one-batch/legacy smoke checks and the new
label, span, side-data and integrated v2 fixtures. Scientific-library warnings
come from deterministic/small synthetic series (ADF and higher moments); the
fixtures do not establish strategy profitability or probability calibration
on real market outcomes.

Run the full suite from a checkout with its project environment:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q --tb=short
```

The working branch used the existing project virtual environment at
`/home/pranay5255/Documents/ethpredict/ETHPredict/.venv/bin/python`.

Focused cross-file verification:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_label_geometry.py tests/test_exact_validation.py \
  tests/test_side_data_integration.py tests/test_staged_trial.py \
  tests/test_forecast_benchmark.py -q
```

## Important assertions

- Both label definitions, both sides, expiry profit/loss, dual-touch ordering,
  missing/unfinished paths, non-candidate exclusion and persisted fields.
- Zero train/evaluation span overlap, inclusive endpoints, irregular clocks,
  mixed timestamp resolutions, held-out outcome mutation invariance,
  per-scenario metrics and deterministic predeclared scenario selection.
- All six side groups enter the actual model matrix. Funding published after
  a candle close cannot enter that candle's features. Future value mutation
  does not change earlier features. Sparse/latest/conflicting sources fail.
- Missing early training coverage cannot be hidden by good overall coverage.
  Disabled side groups preserve baseline matrix/targets; group configuration
  order does not change matrix order; raw side-file changes alter hashes.
- Two-trial time-bar and volume-bar integration fixtures run the main v2
  pipeline with `final_test_guard.evaluate: false` and `benchmark.enabled:
  false`. They assert no final-test query, no test prediction tables, empty
  test metrics, retained split/coverage/label evidence, and explicit disabled
  Trackio receipts. Local artifacts/accounting remain available.
- Source-clock activity bars do not receive a five-minute annualized Sharpe.

## Limits and next steps

Grouped scenarios are independently fitted held-out group combinations, not
stitched CPCV paths. All scenario candidates and metrics are retained; the
summary uses the first declared scenario for each observation. Rank stability
is unavailable until sufficient selected final-test evidence exists.

Side sources remain disabled in the default config. Book/flow latest snapshots
are not usable historical series. The availability contract must be supported
by source evidence; passing a coverage check does not establish that contract.
Activity bars remain OHLCV aggregation proxies with configured bar-count
windows/horizons. Nonzero static hourly funding is refused on activity clocks
until elapsed-time execution costs (#20) exist.

Trackio schema/receipt behavior is covered by tests. The new validation-only
pipeline fixtures explicitly disable delivery; this change does not claim a
new live Trackio delivery/readback run or an external model download.

No new performance experiment, feature promotion, checkpoint promotion or
signed trading operation was performed. The archived #27/#28 results remain
historical evidence. Experiments #31/#32 and later comparisons should wait for
the remaining research controls and declared freeze/selection protocol. The
next implementation issue in the roadmap is #12 (training discipline).
