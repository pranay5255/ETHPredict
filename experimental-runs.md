# Local end-to-end experiments

The v2 pipeline now supports bounded search (#16), purged feature diagnostics
(#17), execution/cost assumptions (#20), and an offline interactive dashboard.
PR #50 supplies the merged training/calibration foundations. Fixed-notional
sizing is the baseline; #19 and full benchmark parity #22 remain separate work.

## Run a frozen first batch

From the repository root, with the existing Lighter candle and funding archive:

```bash
uv run python -m src.experiments.local_batch \
  --source-data data \
  --output-root artifacts/first_batch \
  --prepare-only

OMP_NUM_THREADS=1 uv run python -m src.experiments.local_batch \
  --prepared-config artifacts/first_batch/frozen_config.yml \
  --device cuda:0
```

Preparation creates immutable copies under the chosen output directory and
refuses to overwrite an existing study. It selects a fixed chronological window,
reserves feature warmup, validates a complete five-minute source clock, repairs
only missing hourly funding records through public read-only requests when the
recipe explicitly enables that, and records hashes and retrieval provenance.
The source archive is unchanged. Before training, the runner checks the frozen
config and file hashes again. Review `freeze_manifest.json` and `preflight.json`
where available. A different date window or budget is a different study and must
be declared before examining its final-test outcomes.

`configs/first_experiment_batch.yml` declares four configurations: linear,
8-unit LSTM, 16-unit LSTM, and a regularized LSTM. The seed and order are fixed.
Only six training epochs are allowed, with purged internal early stopping.
The source has 8,192 bars, the first 2,016 support feature warmup, and the model
uses 48-bar sequences. No automatic second search is launched after seeing the
results.

The first recipe enables logistic meta models, training-only Platt calibration,
a training-holdout F1 threshold sweep, and a validation composite that rewards
standard 0.5-threshold meta F1, penalizes ECE, and includes edge-rank usefulness.
It requires 30 validation trades. This classification-oriented selector is a
research hypothesis, not an economic profitability guarantee. Saved threshold
tables distinguish the standard classification score from the fitted trade
threshold. Alternate metric winners are reported without scoring them all on
the final test.

Final test is evaluated only for the predeclared trade-qualified winner, once.
No winner meeting the validation criterion means no final-test read. The ledger
is written before prediction; do not reset it to repeat an evaluation. Inspect
saved results or rebuild the dashboard without retraining:

```bash
uv run python -m src.experiments.dashboard \
  artifacts/first_batch/runs/<run-id>/manifest.json
```

## Open the dashboard

The runner prints the dashboard path. It is a self-contained HTML file with an
embedded Plotly runtime and data. It opens directly from disk; a local server is
convenient when browsing multiple runs:

```bash
uv run python -m http.server 8767 --bind 127.0.0.1 \
  --directory artifacts/first_batch
```

Open `http://127.0.0.1:8767/runs/<run-id>/report/dashboard.html`.

Use the trial selector and validation/final-test selector to inspect:

- Every attempted configuration, status, selection role and executed budget.
- Realized-exit equity, drawdown, return distribution, Sharpe, PSR, DSR,
  underwater durations, concentration, independent-bet proxy and holding times.
- Forecast errors, calibration, rank correlations and expected/realized edges.
- Meta calibration and validation threshold precision/recall/F1, coverage,
  executed trades and simulated PnL. Final test never gets a threshold sweep.
- Base/meta permutation importance, fold variability, trained noise controls,
  separate single-feature fits and configured family-removal ablations.
- Actual fee/spread/slippage/funding decomposition and 1x/2x/5x validation cost
  stress; per-trade fill, delay, cost-model ID and threshold provenance.
- Frozen source/config/code identities and paths to complete JSON/parquet files.

An unavailable statistic retains its reason. A skipped or failed trial remains
in the table. No-trade outcomes remain visible and are not silently replaced by
new attempts. Tables show up to 1,000 matching trades; parquet files contain the
complete set. Charts and metrics remain local. Optional Google Fonts have local
font fallbacks; chart data and the Plotly runtime need no network.

## Search and selection controls

`search.mode`: `grid`, `shuffled_grid`, `random`, `log_uniform`. Discrete domains
remain lists under `search.spaces`; continuous distributions use explicit
`distribution`, `low`, and `high` fields. Supported distributions are `uniform`,
`integer`, and `log_uniform`. Explicit named trials support grid/shuffled grid.
`search.seed` overrides the experiment seed. `max_trials` is an attempt budget,
including failures, skips, repeats and no-trade results, without replacement
attempts. `search_plan.json` stores the complete reconstructable declaration and
executed subset. Large Cartesian spaces retain domains/index reconstruction
rather than allocating the entire population in memory.

Selection aliases: `validation_f1`, `negative_log_loss`, `calibration_error`,
`edge_rank_correlation`, `net_pnl`, and `composite`. Use `selection_mode: min` for
ECE and `max` for the other aliases. Explicit validation metric paths also work;
final-test and non-validation selectors are rejected. Missing scores are ranked
last for either direction and cannot authorize a test evaluation. Model and
threshold comparisons plus feature refits are disclosed in a conservative
comparison budget for DSR; dependent comparisons are not claimed independent.
PBO aligns recorded common period timestamps rather than assuming equal row
counts imply the same observations. Full #38 search-mode comparisons remain a
separately frozen experiment.

## Feature diagnostics

`feature_importance` controls enabled state, seed, repeat count, evaluated folds,
predeclared base columns, SFI and family-removal groups. Noise controls are added
to actual training matrices and candidate features. They are not unused columns
with guaranteed zero importance. Permutation scores are F1/log-loss drops on
purged outer validation; SFI/ablation refits use the exact retained training IDs.
Groups must appear in `features.include`. Positive drops are candidates for
further frozen ablation; noise outranking real inputs or unstable folds is a
rejection flag. Tree impurity importance is descriptive and optional.

## Costs and execution

`costs.mode`: `static`, `stress`, `dynamic`. The shared cost model quotes expected
costs using information available at the signal bar. Realized costs use actual
entry/exit timestamps. These remain separate fields: realized funding and exit
spreads never become meta-model inputs. Labels and backtests retain matching
configuration/source IDs; changing the cost config requires regenerated labels.

Dynamic funding consumes explicit immutable historical files with declared rate
units (`fraction`, `percent`, `bps`), direction semantics and maximum gap. Positive
canonical rates mean longs pay. Publication delay affects the known-rate quote;
settled payments during `(entry, exit]` determine realized funding. The local
Lighter archive uses percent magnitudes and payer direction; this is explicit in
the recipe and is consistent with the archive's value/price relationship.
References: [Lighter funding rules](https://docs.lighter.xyz/trading/funding) and
[the NautilusTrader Lighter adapter](https://nautechsystems.github.io/nautilus_docs/latest/integrations/lighter/#funding-rates).

Spread can use coverage-approved historical order-book data or an explicitly
controlled high/low proxy. Volatility, spread, size and depth can parameterize
slippage. The first recipe uses a proxy and conservative configured fees; it
makes no claim to reproduce a particular account's fee tier or order-book fills.

Execution supports taker, maker haircut, hybrid, delayed-bar entry, deterministic
sampled misses, or expected-fill haircuts. Maker simulations credit no spread
capture and do not calibrate queue position/adverse selection. Delays extend
label spans before purging. Static hourly carry on activity bars uses elapsed
source time. Equity recognizes realized exit cashflows, not signal-time future
profits. These are simulations, not live-execution-ready models.

Cost-stress tables freeze candidates, probabilities, policy and exit outcomes,
and apply PnL-only haircuts. They do not pretend to retrain the supervised label
problem. A full comparison of different cost/entry regimes must regenerate
labels and refit meta models; #40 remains that follow-up study.

## First completed batch and reporting recovery

The first real GPU batch and its interpretation are recorded in
[first-experiment-results.md](first-experiment-results.md). Its source/config
hashes, four validation results and the one final-test result are preserved in
[the compact snapshot](context/first_bounded_backtest_20261016.json).

If a process fails after outcomes are saved but while logging/rendering reports,
recover the reporting stage from its manifest instead of restarting training:

```bash
uv run python -m src.experiments.resume_reporting \
  artifacts/first_batch/runs/<run-id>/manifest.json
```

Recovery protects model, prediction, candidate, test-trade and ledger hashes,
preserves the original manifest, reconciles local Trackio readback and emits a
recovery receipt. It does not fit or predict, reset the ledger, or authorize a
new test evaluation. An already completed receipt makes it idempotent; changed
protected outputs cause an error. Dashboard-only rebuilds can be repeated safely
from saved outcomes. Numerical evidence gates and the declared first-study scope
prevent mere availability of DSR/PBO from being called an alpha result.
