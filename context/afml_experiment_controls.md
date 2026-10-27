# AFML Experiment Controls

> Status note (2026-10-17): this document preserves its original review, protocol
> or historical results; it does not describe the newly collected lake as an
> experiment input. Use [current project status](../PROJECT_STATUS.md) and
> [the lake runbook](lighter_data_lake.md) for current code/PR boundaries,
> coverage, unresolved funding gaps (#54) and next steps.

Use `configs/config.yml` as the baseline v2 research spec. Freeze model, labels,
validation, costs, search budget, and alpha policy before varying one AFML control.

## Bar Clock

Keep the baseline time bars:

```yaml
bars:
  type: time
  time_interval: 5m
```

Compare against activity-driven bars by changing only the bar block:

```yaml
bars:
  type: volume
  threshold_volume: 50000
  min_rows: 256
```

or:

```yaml
bars:
  type: dollar
  threshold_usd: 100000000
  min_rows: 256
```

The run artifacts record raw file hashes, bar coverage, timestamp gaps,
missingness, serial correlation, volatility distribution, and label balance.

## Side Data

Side-data joins must stay disabled unless a join reports enough coverage under the
configured timestamp tolerance. Use `join_side_data_asof` and
`validate_side_data_coverage` from `src.data.features_all`; failed coverage should
block feature use instead of silently forward-filling sparse data.

## Feature Families And Fracdiff

`features.include` is the authoritative v2 feature-family selector. Vary only this
list for feature ablations. `features.frac_diff_mode: fixed_width` is the
reproducible AFML-oriented baseline; `fixed` retains the legacy truncated
convolution entry point and `none` disables the transform. Both convolution
modes now derive width from order and threshold, independently of dataset size.
Short histories stay unavailable until that width is reached (the model matrix
fills warmup NaNs with zero).

Selected `frac_diff_mode: auto` is rejected because fitting its order on the
complete dataset would use future validation/test observations. To compare an
automatically selected order, call `DataPreprocessor.find_optimal_d` on a frozen
training-only calibration series, then write that order into `frac_diff_order`
with `fixed_width` or `fixed` before evaluation. Refit separately per training
fold when studying adaptive orders; the runner does not do that automatically.
An excluded fracdiff family is marked disabled and never fits an order. Missing
feature config uses all baseline families with fixed-width order zero; empty
or unknown selections fail instead of silently changing the input matrix.

CUSUM flags use expanding historical mean and standard deviation. Volatility
regimes use rolling volatility and expanding historical quantiles, with neutral
warmup until two valid volatility observations. These remain AFML-inspired
proxies. Observation sampling uses the separate `sampling.mode` configuration.
Label volatility warmup uses a fixed 0.001 fallback instead of a full-dataset
mean. Appending or changing later observations cannot change historical model
features or that volatility warmup.

`feature_manifest.causality` identifies the observation-time transform scope,
the training-window scaler scope, and full-frame diagnostics that are used only
for reporting. ADF and memory diagnostics are never inputs to feature fitting.
Regression tests recompute complete feature frames on short and long prefixes,
append a different future regime, and verify scaler invariance for every fold
and the final development fit. Old experiment reports remain historical
evidence; the causal transforms change feature values and require new runs
before comparing performance or promoting any feature family.

## Observation Sampling

Dense rolling samples remain the baseline:

```yaml
sampling:
  mode: dense
```

Event-sampled runs can use:

```yaml
sampling:
  mode: cusum
  cusum_threshold_bps: 10
```

or:

```yaml
sampling:
  mode: volatility_cusum
  volatility_multiplier: 1.0
  min_threshold_bps: 1.0
```

Edge-triggered observation sampling can use a feature or prediction column already
present in the feature frame:

```yaml
sampling:
  mode: edge_triggered
  edge_column: log_return
  edge_threshold_bps: 10
```

Candidate trade filtering remains a later policy step after forecasting.

## Frozen Rerun Comparison

After two frozen reruns, compare the run directories or their `manifest.json` files:

```bash
uv run python -m src.experiments.reproducibility artifacts/runs/run_a artifacts/runs/run_b
```

The comparison checks config hashes, raw data hashes, trial count, trial IDs, split
manifest hashes, and metric variance.

## Final-Test Reuse And Run Identity

Trial selection reads validation metrics only. `pipeline.selection_metric` paths
that contain `metrics.test` are rejected before any trial is trained. After
selection, the test split is backtested only for the roles in
`research.final_test_guard.evaluate_selection_roles` (default `raw_best` and
`trade_qualified_best`). Each of those evaluations appends one entry to the
final-test ledger. Unselected trials keep an empty `metrics.test` and do not
touch the ledger.

The ledger key is a hash of the research specification plus the split hash and
an evaluation scope (`meta_label_trial` or `forecast_benchmark`). The research
specification is data, bars, features, sampling, sample weights, targets,
labels, model, training, validation, costs, alpha policy, selection policy, and
seed. `pipeline.run_name`, `pipeline.artifact_root`, experiment labels, and
tracking settings are not part of the hash, so renaming a run is still reuse.
Set `research.final_test_guard.do_not_reuse_test: true` to refuse a second
non-smoke evaluation of the same specification. Pin `ledger_path` when runs
must share a ledger across different artifact directories. Smoke runs increment
the counter and do not block.

`research.raw_data_guard.on_change` controls what happens at run start when the
raw-file hashes differ from the last run of the same research specification:
`warn` (default), `fail`, or `ignore`. Smoke runs record a warning instead of
failing. The comparison result is stored on `run_identity.raw_data_guard`.

Every v2 `feature_manifest` includes `code_identity`, a hash of
`src/data/features_all.py`, `src/features/labeling.py`, and
`src/features/sample_weights.py`, plus a `family_identity_hash` of the configured
feature families, columns, and fracdiff mode. Git summaries list the commit,
dirty state, and critical untracked or modified paths. That critical set
includes `configs/`, `src/`, `tests/`, `context/`, `scripts/`, `TASKS.md`,
`pyproject.toml`, and `uv.lock`.

The forecast benchmark writes `split_manifest.json` with the purged
walk-forward boundaries it actually used (development, gap, test, and each
fold). Each benchmark model that reads the test split increments the same
final-test guard under the `forecast_benchmark` scope before its test metrics
are computed.

## Backtest Statistics And Evidence Grade

Validation and test summaries report a single path (`path_type: single_path`).
Per-period Sharpe is the mean period return divided by its sample standard
deviation. Annualised Sharpe multiplies that by `sqrt(105192)`. The factor is
a 5-minute 24/7 clock: 365.25 days times 288 bars per day. It is not a 252-day
equity factor.

PSR is the Bailey and López de Prado normal CDF. Skewness is the sample
skewness and kurtosis is non-excess (a normal distribution has kurtosis 3), so
the variance term uses `(kurtosis - 1) / 4`. DSR is that PSR at the
Euler-Mascheroni expected-maximum Sharpe. The null mean is zero. The hurdle
uses the recorded trial count and the variance of the trials' per-period
Sharpes. DSR is `unavailable` with a reason when the trial count is below 2 or
that variance cannot be computed.

PBO is a separate CSCV estimate on the validation trial-by-period return
matrix. A split is overfit when the in-sample winner's out-of-sample relative
rank is below the median (negative logit), even if that out-of-sample mean is
still positive. Its `path_type` is `multi_path`. When the matrix is too small,
PBO is `unavailable` with a reason. Full CPCV is not part of this control.
The evidence grade reads each evaluated spec's ledger count after every
selected test, so a later selected trial of the same spec counts as reuse.

`evidence_grade` is `alpha_claim_grade` only when the selected test has at
least `max(min_validation_trades, 1)` trades, DSR is available, the final-test
ledger count is exactly one, and the run is not smoke. Every other run is
`debugging_only`. `failure_modes.concentrated_pnl` uses the positive-return
HHI (default threshold 0.5), not trade coverage.

## Auditable meta labels (#13)

The default label geometry remains `asymmetric`, `success_mode` remains
`positive_net_at_vertical`, and `intrabar_ordering` remains `stop_first`.
Set `geometry: symmetric` for equal cost-adjusted widths, `success_mode:
profit_take_only` for strict target-touch success, or `intrabar_ordering:
profit_first` / `exclude` for explicit alternative OHLC ambiguity assumptions.
Both multipliers must be finite and positive. Candidate tables retain entry,
volatility, barrier return/price levels, planned expiry, actual first touch,
cost/config identity, ambiguity, and exclusion reasons. Missing/invalid OHLC
and unfinished paths remain unlabeled; an observed early touch completes a
label even when its planned expiry has not arrived. No missing prices are
backfilled. Diagnostics report class balance, classification/calibration,
threshold coverage, and ambiguity separately from simulated PnL.

## Exact spans and nested meta validation (#15)

The active config uses `validation.method: exact_span_walk_forward`. Alternatives
are `purged_kfold` and `grouped_scenario`; the latter holds out configured
`scenario_test_groups` combinations of `folds` contiguous development groups.
These are genuine independently fitted grouped scenarios, not stitched CPCV
paths. `max_scenarios` bounds combinatorial work. Each scenario has simulated
validation metrics; reports retain all scenario candidates and PnL quantiles.
The aggregate candidate table uses the first predeclared scenario per row,
never an average of probabilities for different proposals.

Spans are inclusive and use source timestamps (including irregular activity
bars). All supervised horizons and configured meta vertical barriers protect
the final-test boundary. Embargo modes are `fixed_bars`, `max_horizon`,
`sequence_plus_horizon`, `realized_t1`, and `volatility_adaptive` (estimated on
the proposed training population only). Split manifests record retained,
purged and embargoed row IDs and zero-overlap checks. Undersized partitions
fail rather than fall back to an unprotected split. The legacy public
fixed-gap splitter remains; v2 `purged_walk_forward` applies an additional
recorded span safety guard when its configured gaps are insufficient.

For every outer evaluation partition the meta model trains on nested base OOF
candidates generated strictly inside the outer training population. This
prevents indirect label leakage through another outer fold's base model.
Inner folds and any unavailable fallback are recorded in `meta_validation.json`.
Training weights and normalization are recomputed on retained training rows.
Model and calibration enhancements (#12/#18) must preserve these boundaries.

`research.final_test_guard.do_not_reuse_test` is enabled in the active config.
Use `research.final_test_guard.evaluate: false` plus `benchmark.enabled: false`
for implementation-only verification: no test scoring or ledger query occurs.
The forecast benchmark explicitly refuses this mode because it is a test-
evaluating comparison. Rank stability is reported only for actually evaluated
selected trials, and is unavailable when there is insufficient evidence.
Cross-scheme performance comparisons remain the separate frozen #31 experiment.

## Historical side-data contracts and bar coverage (#8)

The baseline still uses time bars with no side data (`data.side_data.groups: {}`).
Enable only explicit historical files. This example describes the existing
hourly funding export; its timestamps must represent actual availability:

```yaml
data:
  side_data:
    groups:
      funding:
        historical: true
        paths: [lighter/fundings_ETH_perp_1h_20260328_20260628.csv]
        timestamp_column: timestamp
        available_at_column: timestamp
        columns: {rate: funding_rate}
        tolerance: 1h
        min_source_rows: 24
        min_coverage: 0.99
        min_split_coverage: 0.99
        fill_policy: zero_with_indicator
```

Alternatively declare a nonnegative `availability_delay` from event time. For
mark candles whose timestamp is the candle open, use at least their actual
completion/publication delay; daily aggregate OI or exchange volume needs its
own end-of-period publication contract. Numeric timestamps require an explicit
`timestamp_unit` (`s`, `ms`, `us`, `ns`). Units are normalized before joins.
Mapping keys are source columns and values are canonical model columns.
The caller is responsible for the truth of a declared availability contract;
a timestamp field or a passed coverage gate alone does not establish it.

Canonical input groups:

- `funding`: `funding_rate` (native `rate` can be mapped).
- `mark_basis`: `mark_price` (native `close` can be mapped); adds mark/close basis.
- `open_interest`: `open_interest` (native `data` can be mapped).
- `exchange_volume`: `exchange_volume` (native `data` can be mapped).
- `order_book`: `bid_price`, `ask_price`, `bid_depth`, `ask_depth`; adds spread
  in basis points and depth imbalance.
- `trade_flow`: historical `buy_volume`, `sell_volume` aggregates; adds imbalance.

Book/flow inputs must be genuine historical aggregates with availability times.
The collected `*_latest.csv` files cannot satisfy this contract. There is no
claim that the repo's current book/trade snapshots provide historical coverage.
Paths are relative to `data.dir` unless absolute; supported formats are CSV
with headers and Parquet. Unknown groups, conflicting duplicates, invalid
availability, negative delays, and latest snapshots fail explicitly. Identical
duplicates are removed and counted; source order and hashes are recorded.

Joins are backward, bounded by `tolerance`, and anchored to each completed
bar's close time. `fill_policy: reject` disallows any missing observation.
`zero_with_indicator` permits zero filling only after the declared global and
per-split coverage gates pass and always adds a group missingness feature.
No future, unbounded, or silent filling is offered. Source/feature families,
column order, raw side hashes, lag and missingness evidence appear in manifests.
Side-group selection is controlled by `data.side_data.groups`; `features.include`
continues to select base OHLCV-derived families independently.

`bars.max_missing_fraction` gates source candle intervals (defaults to zero for
activity bars). Conflicting raw duplicates/invalid OHLCV fail; identical duplicates
are audited. `bars.min_rows` gates the resulting completed bars and
`bars.min_rows_per_split: {train: 32, validation: 16, test: 16}` gates actual
sample populations before fitting. Source/aggregation coverage and incomplete
terminal-bar exclusion are recorded. Group coverage is checked on the union of
actual sequence input windows for every outer and inner partition, not inferred
from the full dataset. Coverage failures carry structured reports.

Volume/dollar bars are explicitly OHLCV aggregation proxies, not reconstructed
trade-tick bars. Incomplete terminal groups are excluded; dollar thresholds use
summed constituent dollar volume. Persisted `stage0_features/bars.parquet`
contains the price path even if OHLCV columns are excluded from the model matrix.
Rolling windows/horizons remain configured bar counts; elapsed time changes on
activity clocks. Annualised Sharpe is unavailable on activity clocks instead of
applying the nominal five-minute factor. This does not add dynamic execution or
holding-period funding (#20); use zero static funding for activity-clock audits
until actual holding-time execution costs are implemented.

Freeze all downstream features/labels/model/validation/costs/policy/seeds when
varying only a clock; freeze the clock and change one side group when studying
joins. Those performance comparisons remain experiments after fixture gates pass.

Explicit `data.bar_type` overrides the baseline `bars.type`; its thresholds and
coverage controls still come from `bars`. Label first-touch timestamps identify
source bars; OHLC inputs do not provide an observed intrabar touch time or order.
Meta audits additionally apply actual realized `t1` intervals to candidate rows,
including a reconstructable candidate-key map and all removal decisions.
