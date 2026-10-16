# Verification: search, importance and execution costs (#16, #17, #20)

Date: 2026-10-16. Base: merged PR #50, `7e43335`.
Branch: `codex/afml-search-importance-execution`.

## Requirement coverage

### #16: bounded search and validation selection

`src/config/search.py` constructs reconstructable seeded grid, shuffled-grid,
random and log-uniform plans. Attempts, including failures, skips, duplicate
proposals and no-trade results, consume the budget without replacement. Plans
save domains, proposal order, executed values and IDs; large discrete spaces use
index reconstruction instead of materializing every combination.

`src/evaluation/selection.py` scores held-out meta F1, negative log loss,
calibration error, edge rank, net PnL and configured composites. Main selection
accepts validation-only metrics, ranks missing values last, records raw/trade-
qualified/classification/calibration winners and alternative metric winners,
and disallows final-test evaluation when the score is unavailable. Existing
accounting/Trackio retain every attempt and the final-test ledger. Conservative
within-run comparison budgets include nested threshold/model comparisons and
importance refits; they are explicitly dependent counts.

Tests cover seeds, bounds, grid expansion, large-space reconstruction,
missing/no-trade/low-trade selectors, status accounting and cross-stage artifacts.
The separate full search-mode/selector comparison remains #38.

### #17: purged importance and trained noise controls

`src/evaluation/feature_importance.py` reports base and meta permutation F1/
log-loss changes only on held-out purged outer validation rows. Single-feature
fits and family removals refit on the exact retained parent training IDs.
Ablation groups must match `features.include`. Noise controls enter the actual
training and candidate matrices; they are not inert dummy columns. Optional tree
impurity importance is descriptive. Positive, negative/redundant, unstable and
noise-control summaries, fold/repeat variation, noise outranking flags, JSON,
Markdown and parquet reports feed artifacts and Trackio.

Known-signal/noise synthetic tests, purged ID checks and family schema tests
passed. Actual noise flags are retained even where they discredit the model.
A first small diagnostic batch does not complete the full feature study #37.

### #20: shared costs and execution assumptions

`src/simulator/costs.py` supplies the same explicit model to candidate filtering,
labeling, forecast edge diagnostics and historical PnL. Expected quotes use only
signal-time state; realized exit spread/funding live in separate fields and do
not enter meta features. Modes cover static, stress and dynamic costs; supported
inputs include declared historical funding units/sign/publication delay,
historical book spread or a declared high/low proxy, and volatility/spread/
size/depth slippage. Coverage failures stop dynamic sources. Source content and
configuration participate in model/raw-data identity. Cost changes without
regenerated labels are rejected.

Taker/maker/hybrid modes, expected haircuts or deterministic sampled misses,
delayed entry, elapsed-time carry and actual completed-bar cashflow clocks are
recorded in labels/trades. Entry delays extend event spans before purging.
Size-sensitive quotes use executable capped fixed notional consistently. PnL
includes component costs, turnover, break-even slippage and return on actual cost
outflows. No maker spread capture or calibrated queue/adverse-selection claim is
made. These are simulations; the first study is not live-execution-ready.

Cost stress freezes validation candidates, probabilities, policy and exit
outcomes and applies 1x/2x/5x PnL haircuts. It explicitly does not claim a
supervised relabel/refit. The full execution-regime comparison remains #40.
Tests cover shared label/PnL cashflows, signed funding timing and units, strict
source gaps, delays/fills, static/stress models, source hashing, cost-config
mismatch and a capped-notional candidate→label→PnL integration.

## Local experiment and dashboard

`configs/first_experiment_batch.yml` and `src/experiments/local_batch.py` freeze a
bounded chronological sample with warmup, immutable source/config hashes and an
attempt budget. Preparation refuses overwrite, repairs only exact missing
funding timestamps when explicitly enabled, preserves source archives and saves
request/response provenance. Hashes are checked again before training.

The real four-trial run used clean training commit `6aeb87c`, an RTX 4090,
seed 17, six epochs maximum, two purged outer folds and fixed 1,000 notional.
No smoke overrides applied. See root `first-experiment-results.md` and tracked
`context/first_bounded_backtest_20261016.json` for values and provenance. All
validation outcomes were negative after costs. The predeclared composite winner
alone was tested: 103 trades and -136.173784 net PnL. There is no alpha claim,
checkpoint promotion, variable sizing, TimesFM parity or current-state inference.

`src/experiments/dashboard.py` builds a standalone embedded-data/Plotly dashboard
from saved outcomes. It covers attempts, budget, equity/drawdown, concentration,
PSR/DSR/PBO limitations, forecast and meta calibration, validation thresholds,
base/meta importance, SFI/family removal, cost stress, filtered trades and frozen
identities. Nonselected test partitions remain unavailable; no test threshold
sweep is shown. Tables cap the displayed trade rows at 1,000; full parquet stays
on disk. Google Fonts are optional; chart runtime/data need no network.

## Reporting recovery and evidence correction

The original process completed all fits, predictions, final-test scoring and
statistics, then exited during Trackio CLI readback because the CLI was outside
PATH. The logger now falls back to the executable next to the running virtualenv
Python. Reporting-only recovery preserves the original manifest, reconciles
already written records, completes logging/rendering and verifies immutable
output hashes. Completed receipts are idempotent and reject changed outputs.
Tests explicitly forbid fitting, predicting and final-test scoring during
recovery and test tamper detection.

The old evidence grade inferred alpha from statistics being available even for
losing outcomes. `src/evaluation/evidence.py` now checks numerical economic and
statistical gates, the explicit claim-study scope and test count. Smoke remains
debugging-only; this bounded study is exploratory. The historical original grade
is retained as legacy evidence rather than silently erased.

Final read-only audit verified:

- Frozen config/candle/funding hashes still match the preparation manifest.
- All **16** protected model, OOF/test prediction/candidate/trade and ledger files
  match reporting-recovery hashes.
- Original validation/test metrics, selection, importance and cost-stress results
  are unchanged by reporting work.
- Exactly **one** nonblocked final-test ledger entry, for `lstm_8`; no other trial
  was tested. Recovery added zero fits, predictions or test evaluations.
- All **six** local Trackio runs were read back: four trials, summary, accounting.

Full checkpoints/data/HTML and `completion_audit.json`, `reporting_recovery.json`,
`dashboard_verification.json` remain in the primary checkout's ignored artifact
root `artifacts/first_bounded_16_17_20/`. The tracked compact snapshot records the
hashes, selected metrics and audit results without checking in large artifacts.

## Software and browser checks

Final suite: `OMP_NUM_THREADS=1 <venv>/bin/python -m pytest -q`:
**218 passed, 21 warnings in 114.29s**, including existing GPU smoke checks.
Warnings concern synthetic constant-series statistics. Targeted final
cost/recovery/dashboard checks also passed (12 tests). `git diff --check` passed.

No connected browser surface was available, so local Google Chrome headless was
used to render and exercise the artifact. Both light and dark palettes passed
all eight trial×partition views, KPI-to-saved-metric equality, equity-to-trade-PnL
sums, unavailable nonselected tests, absence of test threshold rows, trade
filtering, and 1440/390 pixel viewport overflow checks. No JavaScript errors were
captured. Light desktop, dark mobile and the dark feature-diagnostics section
were rendered and visually inspected. The final HTML/template hashes and check
results are in the compact snapshot. Screenshots and a standalone dashboard copy
are under `~/.agent/diagrams/`.

## Remaining work

Merging this batch closes #16/#17/#20, leaving 17 issues: #19 sizing, #22 benchmark
parity, #43–#45 current-state products, and twelve frozen experiments #30–#41.
Bounded fixed-notional historical studies can now proceed; individual comparison
studies still require their declared prerequisites and untouched test policy.
