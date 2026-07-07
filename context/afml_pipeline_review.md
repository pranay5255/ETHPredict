# AFML Review of the Current ETHPredict Pipeline

This review maps the active ETHPredict v2 pipeline against Marcos Lopez de Prado's
*Advances in Financial Machine Learning* (AFML). It is intentionally critical. The
goal is not to say whether the current pipeline is "good" in the abstract, but to
identify which parts are aligned with financial ML practice, which parts are still
research shortcuts, and what should be frozen or varied before trusting a backtest.

The active pipeline is the v2 path in `configs/config.yml`. `run_staged_trial` detects
the v2 target shape and delegates to `run_meta_labeling_mvp`, so the older GLFT
Stage 1/2 runner is not the primary research evaluator for this config.

Primary implementation references:

- `src/experiments/meta_labeling_mvp.py`
- `src/experiments/forecast_benchmark.py`
- `src/data/features_all.py`
- `src/features/labeling.py`
- `src/config/loader.py`
- `src/utils/trackio_logging.py`
- `configs/config.yml`

AFML reference map:

- Chapter 2: financial data structures, bars, event sampling, CUSUM
- Chapter 3: fixed-horizon labels, triple-barrier labels, meta-labeling
- Chapter 4: sample weights, concurrency, uniqueness, sequential bootstrap
- Chapter 5: fractional differentiation
- Chapter 6: bagging, ensemble variance, observation redundancy
- Chapter 7: purged CV and embargo
- Chapter 8: feature importance
- Chapter 9: hyperparameter tuning with purged CV and financial scoring
- Chapter 10: bet sizing
- Chapters 11-12: dangers of backtesting, walk-forward, CPCV
- Chapter 14: backtest statistics and classification metrics
- Chapters 17-19: structural breaks, entropy, microstructural features

## Executive Assessment

The pipeline has a sensible skeleton for AFML-style research:

- raw data is materialized and tracked;
- features are computed before supervised labels;
- the base forecaster and meta-labeler are separated;
- out-of-fold predictions are used for meta-label training;
- purged walk-forward splits exist;
- cost-aware candidate filtering exists;
- backtest artifacts are written per run;
- benchmark forecasts are separated from trading PnL.

The main methodological gaps are still material:

- The active data clock is fixed 5-minute time bars, which AFML treats as a weak
  default because market information does not arrive uniformly in time.
- The v2 targets still include fixed-horizon next-return prediction before the
  triple-barrier meta layer. That is useful as a benchmark target, but it is not the
  strongest AFML labeling design.
- Sample weights are uniform in the active v2 dataset, even though labels overlap
  and AFML explicitly treats label concurrency as a non-IID problem.
- Purging and embargo are implemented by fixed bar gaps, not by exact label span
  overlap using each observation's realized `t1`.
- Feature importance is not yet part of the research loop, so the project still risks
  using the backtest as a discovery tool.
- The backtest is a single historical path with grid selection. It does not estimate
  probability of backtest overfitting, deflated Sharpe, or CPCV path dispersion.
- Bet sizing is currently fixed-notional and threshold-based, not probability-sized.
- Execution costs are configurable but crude; slippage, queue position, latency,
  spread dynamics, and funding timing are not yet first-class state variables.

The pipeline is good enough for exploratory model debugging. It is not yet good
enough for claims of tradable alpha.

## 1. Data Source and Bar Construction

Current behavior:

The pipeline reads Lighter ETH perpetual OHLCV CSVs from `data/raw`. The active
granularity is `5m`. `DataPreprocessor.load_price_data` dedupes timestamps, sorts
rows, and feeds OHLCV into feature engineering. Lighter-native side data exists under
`data/lighter`, but it is not joined into the v2 model feature frame.

AFML lens:

AFML Chapter 2 distinguishes time bars from activity-driven and information-driven
bars. Its critique of time bars is directly relevant here: time bars oversample quiet
periods and undersample active periods. For a perpetual crypto market, this is not a
minor issue. The same 5-minute bar can represent dead overnight drift or a liquidation
cascade. Treating both as equivalent samples is a strong and usually false assumption.

Deep critique:

The current use of 5-minute time bars is the largest upstream bias in the pipeline.
Almost every later step inherits that clock: realized volatility, entropy, fracdiff,
sequence windows, target horizons, purge gaps, and trade-rate limits. If the data
clock is wrong, later sophistication may only optimize around a poor sampling
choice.

The current raw data is also price/volume only for model features. This is an
acceptable first baseline, but it misses precisely the market microstructure variables
AFML Chapter 19 treats as information-rich: signed flow, spread, depth, cancellation
pressure, order imbalance, and impact. The repo has order book, trade, funding, open
interest, and volume side files, but they are not historical enough or joined enough to
be trusted as live model inputs yet.

Freeze-and-backtest experiments:

1. Freeze model, labels, costs, validation, and policy. Vary only the bar clock.
   Compare current 5-minute time bars against dollar bars and volume bars built from
   available trade data when historical trade depth is sufficient.

2. Freeze the 5-minute clock. Vary only side-data joins:
   no side data, funding only, mark basis only, open interest and exchange volume,
   order-book/trade-flow features where historical coverage is valid.

3. Freeze all downstream settings. Measure return serial correlation,
   heteroscedasticity, and label balance by bar type before training models. A bar type
   that improves distributional properties but hurts backtest PnL should not be
   rejected immediately; it may be exposing that the old PnL was clock bias.

Priority:

High. This is an upstream design decision that affects the validity of every later
stage.

## 2. Feature Engineering

Current behavior:

`features_all.py` computes OHLCV, price returns, log returns, volume changes,
dollar volume, high-low range, rolling return volatility, volume z-score, fractional
differentiated close, return entropy, CUSUM flag, SADF flag, volatility regime, and
Parkinson volatility. The v2 pipeline then normalizes the feature matrix globally
inside `build_multi_horizon_lighter_dataset`.

AFML lens:

AFML Chapters 5, 17, 18, and 19 support several ingredients already present:
fractional differentiation, structural-break statistics, entropy, and high-low
volatility. But AFML is careful about how these are calibrated and evaluated.
Features should preserve memory while achieving stationarity, be tied to an event
clock where possible, and be validated through feature importance before backtest
iteration.

Deep critique:

The current feature set is directionally inspired by AFML, but several
implementations are still shallow proxies:

- `features.frac_diff_order` exists in config, but `_feature_frame` calls
  `find_optimal_d` internally and ignores the configured order. That makes
  experiment reproducibility weaker than it looks.
- The fractional differentiation implementation is closer to an expanding-window
  truncation by threshold. AFML specifically warns that expanding-window
  fractional differentiation can introduce drift, and recommends fixed-width fracdiff
  for stable features.
- CUSUM is a simple flag over the full return series, not event-based sampling. It
  currently adds a feature, but it does not change the sample set.
- SADF is simplified and may be computationally cheap, but the review should not
  assume it is equivalent to AFML's more expensive SADF process.
- Feature normalization appears to use the full feature frame before train/test
  splitting. That can leak future distributional information into earlier samples.
  Even if labels are not leaked, this is avoidable preprocessing leakage.
- Feature selection is not governed by `features.include`; the pipeline always uses
  `LIGHTER_FEATURE_COLUMNS`. That makes advertised feature ablations misleading.

Freeze-and-backtest experiments:

1. Freeze everything except feature families:
   OHLCV only, OHLCV plus volatility, plus entropy, plus fracdiff, plus structural
   break flags, plus side-data features.

2. Freeze feature families. Vary only fractional differentiation:
   no fracdiff, current auto `d`, configured fixed `d`, fixed-width fracdiff with a
   stationary-memory diagnostic.

3. Freeze model and labels. Run purged OOS feature importance:
   permutation importance and single-feature importance on the meta-labeler inputs,
   not just the base model inputs.

4. Freeze all features. Compare global normalization against fold-local
   normalization fitted only on training/development windows.

Priority:

High. Before doing more backtests, add a feature-importance and preprocessing-leakage
audit. AFML would treat this as core research hygiene.

## 3. Event Sampling and Candidate Times

Current behavior:

The pipeline creates a sample for nearly every possible rolling sequence after
`sequence_length`, subject to enough future horizon. Events are not selected by a
CUSUM filter or by information-driven triggers. Candidate signals are generated
later from model predictions.

AFML lens:

AFML Chapter 2 recommends event-based sampling so that labels are attached to
informative events, not arbitrary clock ticks. The CUSUM filter is a way to sample
only when the process has moved enough to matter.

Deep critique:

The current pipeline conflates two concepts:

- when to create an ML observation;
- when to trade after a forecast.

AFML separates them. You first identify events worth labeling, then you label and
model those events. In ETHPredict, every rolling 5-minute sequence becomes a
training observation, while `candidate_signals` later filters predicted edges. This is a
valid dense-forecasting design, but it creates high overlap, strong serial dependence,
and a massive class of low-information samples. It also makes label uniqueness worse.

The current CUSUM flag is not enough to claim event-based sampling. It is a feature
inside dense samples, not a sampling rule.

Freeze-and-backtest experiments:

1. Freeze features, model, labels, and policy. Vary sampling:
   dense rolling samples, CUSUM-triggered samples, volatility-scaled CUSUM samples,
   predicted-edge-triggered samples after a first-stage model.

2. Freeze candidate policy. Compare label uniqueness and effective sample size for
   dense vs event-sampled data.

3. Test whether event-sampled models have lower raw trade count but better
   precision, lower turnover, and better net PnL per trade.

Priority:

High. This could reduce overfitting and computation while making labels more
meaningful.

## 4. Target Construction and Base Forecasting

Current behavior:

The base model predicts future log returns and future direction for configured
horizons. Active horizons are `next_5m` and `next_hour`. Forecast benchmarks
include zero return, momentum, LSTM, and optional TimesFM. The benchmark path
intentionally separates forecast metrics from trading metrics.

AFML lens:

AFML Chapter 3 criticizes fixed-time horizon labels because they ignore path, stops,
and volatility context. Fixed-horizon return forecasting can still be useful as a base
forecast, but AFML would not treat it as sufficient evidence for tradable labels.

Deep critique:

The current base forecasting target is useful but not fully aligned with AFML's
preferred labeling approach. Predicting next-hour return is a regression problem;
trading it is a path-dependent decision problem. A forecast can have low MSE while
being unusable after costs, stops, and slippage. Conversely, a noisy forecast can still
be useful if it ranks candidate opportunities well.

The benchmark design is a strong point. Zero-return and momentum baselines are
needed. TimesFM is also useful as an external zero-shot benchmark, but it should not
be trusted until its context, scaling, and forecast horizon behavior are audited against
crypto microstructure. A large pretrained model can look sophisticated while still
being badly calibrated to local execution realities.

Freeze-and-backtest experiments:

1. Freeze labels and policy. Compare base forecasters:
   zero return, momentum, LSTM, TimesFM, and eventually tree/tabular models.

2. Freeze base predictions. Evaluate multiple mappings from forecast to candidate:
   raw sign, probability sign, predicted edge after costs, volatility-scaled edge, and
   horizon agreement.

3. Freeze all downstream logic. Report forecast quality separately:
   MAE/MSE, directional log loss, calibration, rank correlation of predicted edge to
   realized net edge, and edge-bucket realized performance.

Priority:

Medium-high. The benchmark layer is well-designed, but it should be expanded from
point metrics to calibration and rank quality.

## 4A. Model Training, Regularization, and Ensembles

Current behavior:

The active base model is a shared multi-horizon LSTM with return and direction
heads. It trains for a configured number of epochs using Adam, MSE for return, BCE
for direction, and uniform sample weights. The final LSTM is retrained on the full
development set before test prediction. There is no early stopping, bagging,
sequential bootstrap, or ensemble uncertainty in the active v2 path.

AFML lens:

AFML Chapter 6 frames overfitting as a variance problem and treats bagging as a
variance reducer, but only when the individual learners are meaningfully
decorrelated. Chapter 4 matters here: if observations are highly redundant, naive
bootstraps produce similar learners and out-of-bag accuracy can be inflated.

Deep critique:

The current LSTM path is simple and reproducible, but it has weak defenses against
financial overfitting:

- Dense overlapping samples give the model many near-duplicates.
- Uniform weights make the optimizer care most about crowded market periods.
- A single neural model gives no estimate of model instability.
- There is no early stopping based on purged validation loss.
- There is no bagging or seed ensemble to reveal variance across fits.
- There is no comparison to simpler tabular models on latest-bar and rolling-window
  summary features.

The current low default capacity is conservative, which is good for smoke research,
but low capacity does not solve the non-IID sample problem. A small overfit model
can still be overfit if the data construction leaks redundant labels.

Freeze-and-backtest experiments:

1. Freeze data, labels, split, and policy. Vary only training discipline:
   single LSTM, early-stopped LSTM, seed ensemble, uniqueness-weighted LSTM,
   and bootstrap/sequence-bootstrap ensemble.

2. Freeze model architecture. Vary only capacity:
   hidden size, layers, dropout, weight decay, and epochs. Select by purged validation
   forecast and meta-label metrics before looking at test PnL.

3. Freeze neural setup. Add non-neural baselines:
   logistic/linear models for meta-labels and tree/tabular base models for latest-bar
   features. If the LSTM cannot beat simple models after costs, complexity is not yet
   justified.

Priority:

High. Neural capacity should not be expanded until weighting, early stopping,
and model-instability diagnostics are in place.

## 5. Triple-Barrier Meta-Labeling

Current behavior:

`candidate_signals` converts each horizon prediction into long/short candidate rows
when expected edge clears a configured threshold. `meta_triple_barrier_labels` then
applies side-aware barriers using realized volatility, configured costs, `profit_kappa`,
`stop_kappa`, and horizon bars. Non-candidates stay in artifacts with `meta_label =
NaN` and are excluded from binary meta training.

AFML lens:

This is one of the strongest AFML-aligned parts of the current pipeline. AFML
Chapter 3 uses triple-barrier labels to account for path-dependent exits and uses
meta-labeling when a primary model supplies the side while a secondary model
decides whether to act.

Deep critique:

The architecture is right, but implementation choices deserve scrutiny:

- The primary model is not exogenous in the quantamental sense. It is another ML
  model trained on the same feature domain. That is acceptable, but it means the
  meta-labeler may inherit base-model overfitting.
- The candidate threshold is based on predicted return less configured cost, but
  predicted returns may be miscalibrated. A small calibration error can radically
  change candidate coverage.
- The barrier uses `realized_vol` at the sample. That is sensible, but the current
  volatility estimate comes from time bars and a fixed rolling window.
- Barrier labels are tied to horizon bars. This is simpler than a fully event-driven
  `t1`, but it makes the vertical barrier deterministic rather than dependent on the
  first true event end.
- The vertical-barrier label treats positive net return as a successful meta label even
  without hitting profit take. That is a defensible choice, but it should be compared
  against a stricter "profit-take only" definition.
- The label function checks adverse move before favorable move inside each bar. For
  OHLC bars without intrabar path, this is conservative for ambiguous bars, but it
  is still an assumption.

Freeze-and-backtest experiments:

1. Freeze base predictions and costs. Vary only label geometry:
   profit/stop kappa, symmetric vs asymmetric barriers, vertical positive-net vs
   profit-take-only success, conservative vs optimistic intrabar ordering.

2. Freeze labels. Vary candidate generation:
   min edge, horizon agreement requirement, volatility regime filter, funding filter.

3. Freeze all else. Measure meta-label class balance, precision, recall, F1,
   probability calibration, and trade coverage. Do not optimize only net PnL.

Priority:

High. This stage determines what "good trade" means. If this definition is unstable,
all later PnL is unstable.

## 6. Sample Weights and Non-IID Outcomes

Current behavior:

The active v2 dataset sets every sample weight to `1 / len(X)`. The training loop uses
those weights, but they are uniform.

AFML lens:

AFML Chapter 4 is explicit that overlapping label spans create non-IID outcomes.
It recommends measuring label concurrency, average uniqueness, and using those
weights to reduce the undue influence of redundant observations. It also introduces
sequential bootstrap as a way to sample less redundant training sets.

Deep critique:

This is a major gap. ETHPredict has overlapping samples by construction:

- 288-bar sequence windows overlap heavily.
- `next_hour` labels overlap across adjacent 5-minute samples.
- Triple-barrier meta labels can span multiple bars.
- Dense rolling samples turn one market move into many highly similar examples.

Uniform weighting lets redundant regions dominate training. The model can appear
to learn because it sees many near-duplicates of the same market event. Purging
reduces validation leakage, but it does not fix the training objective's weighting
problem.

Freeze-and-backtest experiments:

1. Freeze data, labels, model, and policy. Vary sample weighting:
   uniform, horizon-span uniqueness, triple-barrier `t1` uniqueness, return-magnitude
   weighting, and combined uniqueness-by-return weighting.

2. Freeze weights. Vary sampling:
   standard shuffled DataLoader, sequential bootstrap batches, chronological batches.

3. Report effective sample size and average uniqueness by run. A model with fewer
   effective samples but better OOS precision is more credible than one with higher
   dense-sample accuracy.

Priority:

Very high. This is one of the clearest AFML mismatches in the current v2 path.

## 7. Purged Walk-Forward Validation

Current behavior:

`purged_walk_forward_splits` makes chronological folds. Each fold trains on earlier
samples, validates on later samples, and reserves a final holdout test set. It subtracts
`purge_bars` and `embargo_bars` before the test/development boundary and between
train and validation windows.

AFML lens:

AFML Chapter 7 recommends purging observations whose label intervals overlap the
test labels, plus embargoing observations immediately after test intervals when
features are serially dependent. Chapter 12 further distinguishes historical
walk-forward simulation from CV and CPCV scenario testing.

Deep critique:

The current split is directionally correct but simplified:

- It uses fixed bar counts, not exact label span overlap.
- It does not compute `t1` per sample and purge based on actual first barrier touch.
- It trains only on earlier data for validation folds, which is historically clean but
  does not produce the same set of scenario tests as purged k-fold CV or CPCV.
- With only a few folds, the model selection surface can still be highly path-dependent.
- The final test split is untouched in code, but repeated human iteration on the same
  final test would still create selection bias.

Freeze-and-backtest experiments:

1. Freeze features, labels, and model. Vary validation scheme:
   current walk-forward, exact-span purged walk-forward, purged k-fold, and CPCV.

2. Freeze split scheme. Vary purge/embargo:
   max horizon only, max sequence length plus horizon, realized `t1` span, and
   volatility-adaptive embargo.

3. Track validation-to-test rank stability. If trial rankings change wildly across split
   schemes, the strategy is probably not robust.

Priority:

High. The current split is a good start, but not enough to support strong claims.

## 8. Hyperparameter Search and Trial Selection

Current behavior:

`expand_grid_search` expands dotted config paths and truncates to `max_trials`.
`select_trials_with_trade_floor` selects the raw best trial by configured metric, then
separately records the best trade-qualified trial if the raw best produced too few
validation trades.

AFML lens:

AFML Chapter 9 allows grid search, but only with purged CV and appropriate scoring.
For meta-labeling, AFML recommends scores that punish degenerate classifiers, such
as F1, rather than naive accuracy. Chapters 11 and 14 warn that every tried
configuration contributes to multiple-testing risk.

Deep critique:

The trade-floor selection logic is a good practical guard. It prevents a no-trade model
from winning merely because zero PnL beats negative PnL. However, trial selection
is still backtest-sensitive:

- The grid is deterministic and truncated, so parameter order can affect what is tried.
- Search is currently tied to validation net PnL by default. AFML would prefer that
  research decisions be made from labels, feature importance, calibration, and
  classification quality before consulting trading PnL.
- There is no accounting for the number of tried configurations in final statistics.
- No randomized/log-uniform search exists for continuous hyperparameters.
- Hyperparameter search does not currently optimize a meta-label specific objective
  such as F1 or negative log loss at the meta layer.

Freeze-and-backtest experiments:

1. Freeze data and split. Compare selection metrics:
   validation net PnL, validation F1, negative log loss, rank correlation of expected
   edge to realized net edge, and a composite requiring minimum trades.

2. Freeze metric. Compare search modes:
   deterministic grid, shuffled grid, random search, and log-uniform search for
   learning rate/regularization.

3. Record every trial in Trackio and report trial count, best raw result, best
   trade-qualified result, and deflated performance estimates.

Priority:

Medium-high. The current machinery is usable, but it can still reward overfit
thresholds.

## 9. Feature Importance and Research Discipline

Current behavior:

There is no active feature-importance stage in the v2 run. Feature and model changes
are evaluated mostly through forecast metrics and backtest metrics.

AFML lens:

AFML Chapter 8 treats feature importance as a research tool and warns against using
backtests as the discovery loop. It recommends methods such as MDI, MDA, SFI,
orthogonalized features, and synthetic controls, with purged CV.

Deep critique:

This is a process gap. Without feature importance, the pipeline can answer "did this
configuration backtest well?" but not "what did the model actually learn?" That makes
it easy to iterate toward false discoveries.

This is especially important because the current feature set contains correlated
families: returns, log returns, close-open return, high-low range, volatility estimates,
Parkinson volatility, entropy, and vol regime. Substitution effects can hide or dilute
true predictors. The meta-labeler also uses model outputs plus market state features,
so feature importance should be run separately for the base forecaster and the
meta-labeler.

Freeze-and-backtest experiments:

1. Freeze a candidate model. Run purged permutation importance on meta-labeler
   features, scored by F1 and negative log loss.

2. Freeze a feature set. Add random noise features and verify importance methods do
   not rank noise highly.

3. Freeze labels and split. Run feature ablations in predeclared groups before any
   test-set backtest.

Priority:

Very high. This should be added before large-scale backtest iteration.

## 10. Meta-Labeler Model

Current behavior:

The meta-labeler is a small MLP over candidate-level features. It falls back to a
constant probability model when the training labels are empty or single-class.

AFML lens:

AFML's meta-labeling idea is model-agnostic. The critical requirement is that the
secondary model learns whether to act on primary positives using out-of-sample
primary predictions. It should improve precision without destroying recall.

Deep critique:

The fallback behavior is good engineering. It prevents crashes and exposes
degenerate data. But the meta model remains under-audited:

- There is no probability calibration check.
- There is no explicit F1/precision/recall threshold selection table in the main
  summary, although diagnostic helpers exist.
- It trains on candidates only, which is correct for binary meta-labeling, but the
  broader no-trade policy is implicit in candidate generation rather than learned.
- A small MLP may overfit sparse candidate rows if class balance is poor.
- Tree models may be a better first meta-labeler because they are easier to inspect
  and more compatible with feature importance.

Freeze-and-backtest experiments:

1. Freeze candidate rows and labels. Compare meta models:
   constant, logistic regression, calibrated tree ensemble, MLP.

2. Freeze meta model. Sweep threshold only on validation:
   report precision, recall, F1, selected trade count, and validation net PnL.

3. Freeze threshold. Run test once. Do not revisit test after threshold selection.

Priority:

Medium-high. The concept is right, but the model needs calibration and
interpretability.

## 11. Candidate Policy and Bet Sizing

Current behavior:

The policy filters by `meta_prob`, expected edge, optional horizon selection, max
trades per day, cooldown, and max turnover. Position sizing is fixed notional capped
by `max_position_notional`.

AFML lens:

AFML Chapter 10 emphasizes that accuracy alone is not enough. Bet sizing should
reflect confidence, concurrent bets, and probability of correctness. Meta-label
probabilities are natural inputs to size.

Deep critique:

The current fixed-notional approach is a good baseline, but it leaves money
management mostly outside the model. It makes all accepted trades equal even if one
has `meta_prob = 0.56` and another has `meta_prob = 0.90`. That makes the threshold
a brittle cliff and ignores probability calibration.

The current design also collapses overlapping active signals by keeping one trade per
timestamp, but it does not average concurrent active bets across holding periods in
the AFML sense. For a high-frequency ETH strategy, concurrency matters because
several horizons can remain active while new signals arrive.

Freeze-and-backtest experiments:

1. Freeze predictions and labels. Vary sizing:
   fixed notional, probability-scaled, edge-scaled, meta-prob times edge, and
   concurrency-budgeted sizing.

2. Freeze sizing. Vary trade throttles:
   cooldown, max trades per day, max turnover, horizon-specific caps.

3. Report turnover, exposure, PnL per turnover, return on execution costs, and
   drawdown, not only net PnL.

Priority:

Medium. Fixed notional is acceptable while alpha quality is unknown, but it should
not be the final policy.

## 12. Cost and Execution Modeling

Current behavior:

Costs include fees, spread, slippage, and funding bps per hour. Candidate filtering,
triple-barrier labels, and PnL all use these costs. Backtest trades assume fixed
notional and label-derived net returns.

AFML lens:

AFML Chapters 11 and 14 warn that transaction costs and implementation shortfall
are common sources of false backtests. The true cost of execution cannot be known
without interacting with the book, but the backtest should stress realistic worse-case
assumptions.

Deep critique:

Cost modeling is present, which is good. The problem is that the same static
cost assumptions affect candidate generation, labels, and PnL. A cost sweep is
therefore a change to the learned problem, not just a PnL haircut. That is fine, but it
must be interpreted correctly.

The execution model is not yet tied to market state:

- spread is static, not from historical top of book or high-low estimators;
- slippage is static, not function of size, depth, volatility, or flow toxicity;
- funding is flat per hour, not tied to actual funding timestamps and rates;
- maker/taker choice is not modeled;
- latency and missed fills are not modeled;
- market impact is not modeled.

Freeze-and-backtest experiments:

1. Freeze predictions and policy. Run cost stress:
   base costs, 2x costs, 5x costs, dynamic funding, dynamic spread if available.

2. Freeze costs. Vary execution assumption:
   taker only, maker only with fill probability haircut, hybrid, delayed entry by one
   bar.

3. Report break-even slippage and PnL per turnover. A strategy that only works under
   optimistic costs should be rejected early.

Priority:

High before live trading. Medium for current research if clearly labeled as simulated.

## 13. Backtest Design

Current behavior:

The v2 alpha backtest evaluates validation and test candidate sets, writes trades, and
reports metrics such as coverage, trades, gross/net PnL, fees, turnover, exposure,
hit ratio, win rate, average win/loss, drawdown, return quantiles, horizon distribution,
and side distribution.

AFML lens:

AFML Chapters 11 and 12 argue that backtests are not research tools and that a
single walk-forward path can be overfit. Chapter 14 recommends broader statistics:
time range, AUM, capacity, leverage, bet frequency, holding period, turnover,
implementation shortfall, drawdown/time-under-water, Sharpe, probabilistic Sharpe,
deflated Sharpe, and classification scores.

Deep critique:

The current backtest is useful as a sanity check. It is not strong evidence of
tradability:

- It is a single realized historical path.
- It does not report a distribution across CPCV paths.
- It does not estimate PBO or deflated Sharpe.
- It does not clearly distinguish independent bets from individual trades.
- It does not report time under water, concentration of returns, or PnL per turnover.
- It does not record enough context to prove that a final result was not selected after
  many unreported experiments, although Trackio can fix this process gap.

The strongest current design choice is separating raw best from trade-qualified best.
That prevents a common no-trade selection failure.

Freeze-and-backtest experiments:

1. Freeze all model decisions. Add richer statistics:
   holding period, bet frequency, HHI concentration, time under water, return on
   execution costs, PSR, DSR, and trial count.

2. Freeze the research specification. Run CPCV or grouped scenario CV and compare
   path distribution to the single walk-forward result.

3. Freeze validation-selected thresholds. Evaluate test once, then archive the run as
   immutable.

Priority:

Very high if the goal is to make claims about alpha. The current backtest is a
debugging tool, not a final evaluator.

## 14. Forecast Benchmark and TimesFM

Current behavior:

`forecast_benchmark.py` evaluates baseline and model forecasts using a common
schema. Each benchmark can optionally route its predictions into the meta-label and
backtest flow. TimesFM is optional and skipped on failure.

AFML lens:

AFML does not discuss TimesFM, but it strongly supports separating model research
from trading backtest. A common benchmark schema is consistent with AFML's
research discipline because it prevents every model from getting a custom evaluator.

Deep critique:

This is one of the better engineered parts of the repo. The weakness is not structure,
but evaluation depth:

- Forecast metrics should include calibration and rank usefulness, not just error and
  direction metrics.
- TimesFM should be evaluated as an external benchmark, not assumed superior.
- The benchmark should use the exact same frozen split and costs as the main
  meta-label flow.
- The zero-return baseline should remain prominent because it catches spurious
  directional claims.

Freeze-and-backtest experiments:

1. Freeze split and policy. Route each benchmark model through identical
   meta-label/backtest logic.

2. Freeze model. Compare forecast-only selection against backtest selection. If the
   best forecaster is not the best trader, inspect why.

3. Track benchmark results in Trackio and HF Spaces before large model changes.

Priority:

Medium. Structurally strong, but needs richer diagnostics.

## 15. Trackio and Experiment Accounting

Current behavior:

Trackio logging is optional and defensive. The code logs benchmark model runs,
meta-label trials, and run summaries. Metrics are flattened and artifacts are attached
by path.

AFML lens:

AFML's backtesting chapters emphasize recording every trial. Experiment accounting
is not just operational convenience; it is required to estimate multiple-testing risk.

Deep critique:

Trackio is correctly placed in the pipeline, but the current metrics logged are not yet
complete enough for AFML-grade audit. The future HF Spaces dashboard should not
only show "best run." It should show the whole trial surface, trial count, failed/skipped
runs, no-trade runs, and how many times the same test set has been queried.

Freeze-and-backtest experiments:

1. Treat Trackio as mandatory for non-smoke runs. Freeze run naming and config
   hashing so every result has an immutable identity.

2. Log all hyperparameters, feature families, split IDs, artifact paths, validation
   metrics, test metrics, and trial count.

3. Build dashboard panels around AFML failure modes:
   no-trade winners, low-trade winners, high turnover, poor calibration, unstable
   validation/test ranking, high cost sensitivity, and concentrated PnL.

Priority:

High. This is the control system that keeps future experimentation honest.

## 15A. Artifacts, Manifests, and Reproducibility

Current behavior:

The pipeline writes `resolved_config.yml`, environment metadata, stage manifests,
prediction parquet files, candidate parquet files, trade parquet files, JSON reports,
and markdown summaries under `artifacts/runs/<run_id>`. The benchmark and
meta-label paths record model-level manifests. The local repo also contains a visual
pipeline explainer.

AFML lens:

AFML backtesting critique depends on auditability. If every data transformation,
trial, parameter choice, and backtest query is not recorded, it becomes impossible to
assess multiple-testing risk or reproduce the selected result.

Deep critique:

The artifact system is a strong foundation, but it is not yet an immutable research
ledger:

- Raw data file hashes are not consistently attached to the v2 run manifest.
- The untracked AFML PDF shows the worktree can contain important local files
  that are not part of reproducible experiment state.
- Config hashes, git dirty state, and exact trial counts should be first-class summary
  fields for every run.
- Feature code version and feature-family switches are not recorded at a granular
  enough level to reproduce future ablations.
- There is no `do not reuse test` guard or counter for test-set evaluations.

Freeze-and-backtest experiments:

1. Freeze a run spec and rerun it twice. Compare manifests, split boundaries, trial
   counts, and metrics variance. Non-determinism should be known and recorded.

2. Freeze raw data. Add raw file hashes to v2 manifests and verify the same run can
   reject changed data.

3. Freeze reporting schema. Make Trackio/HF Spaces consume only manifest fields
   and artifact paths, so the dashboard is an audit layer, not a separate source of
   truth.

Priority:

Medium-high. Reproducibility is already partially implemented, but AFML-grade
backtest accountability needs stricter run identity and trial accounting.

## Recommended Freeze Matrix

Use this sequence to avoid researching under the influence of the backtest:

1. Freeze the data window and bar clock.
   Audit distribution, serial correlation, heteroscedasticity, missing intervals, and
   label balance. Do this before model changes.

2. Freeze feature definitions.
   Run feature importance and ablations using purged validation. Do not use final
   test PnL to choose features.

3. Freeze base forecast candidates.
   Compare zero return, momentum, LSTM, TimesFM, and future tree baselines with
   forecast metrics, calibration, and edge ranking.

4. Freeze candidate generation.
   Choose a predeclared rule for side, edge, and minimum predicted edge.

5. Freeze meta-label definition.
   Choose profit/stop/vertical definitions and intrabar ambiguity handling.

6. Freeze validation scheme.
   Prefer exact-span purging and add CPCV or scenario paths once feasible.

7. Freeze policy and sizing.
   Select thresholds, size rule, trade limits, and cost assumptions on validation only.

8. Run one final test evaluation.
   Archive it. If it fails, start a new research specification rather than tuning against
   the same test.

## Highest-Value Next Implementation Steps

1. Add exact label-span metadata.
   Persist each observation's start time and realized or maximum end time. Use this
   for sample uniqueness, purging, and embargo.

2. Replace uniform v2 sample weights.
   Implement average uniqueness weights for both base training and meta-label
   training.

3. Add fold-local preprocessing.
   Fit scalers only on training windows and apply to validation/test. Avoid full-sample
   normalization.

4. Add feature-family switches.
   Make `features.include` actually control the active feature set.

5. Add fixed-width fractional differentiation.
   Make `features.frac_diff_order` and threshold reproducible config values.

6. Add feature importance reports.
   Start with purged permutation importance and single-feature importance for the
   meta-labeler.

7. Add richer backtest statistics.
   Include holding period, time under water, HHI concentration, PSR/DSR, PnL per
   turnover, and return on execution costs.

8. Add CPCV or at least grouped scenario CV.
   The current walk-forward path is useful but too easy to overfit.

9. Add dynamic cost features.
   Join historical funding, mark/trade basis, spread/depth if available, and stress
   slippage as a function of volatility and size.

10. Build the Trackio/HF Spaces dashboard around failure modes.
    Make the dashboard show why a run should not be trusted, not only why it looks
    good.

## Bottom Line

The current ETHPredict pipeline is a credible prototype of an AFML-style research
stack, especially around triple-barrier meta-labeling, purged out-of-fold base
predictions, benchmark separation, and artifact tracking. The main risk is that the
pipeline can still generate polished false positives because it lacks AFML's strictest
controls: activity-based bars, exact uniqueness weights, feature-importance-first
research, exact purging by label spans, CPCV or scenario-path backtesting, and
trial-count-aware performance statistics.

The right next move is not to add a larger model. The right next move is to make the
research loop harder to fool.
