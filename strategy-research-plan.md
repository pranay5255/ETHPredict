# ETH strategy research plan - 2026-10-17

The immediate goal is to find evidence of positive after-cost expectancy, with a
credible mechanism and a reproducible decision rule. A numerical probability of
future profitability cannot be estimated reliably from the current short study.
The unchanged configuration has negative evidence. The software infrastructure
makes further research possible; it does not itself supply a trading edge.

This document preserves the recommendations made before the bounded development
study. Its forecast repair/opportunity/gate work is now implemented in draft
[PR #53](https://github.com/pranay5255/ETHPredict/pull/53), stacked on open PR #51;
it is not yet merged. The study rejected every configuration: unfiltered rules
lost 71.941–78.851 net; the only positive overlay had two trades and failed 2x
cost stress. No new final test was used. Do not repeat the original study or
reinterpret the recommendations below as a validated profitable strategy.

The mainnet lake now covers five markets from June 2025 through October 16, 2026,
but the general training pipeline still reads CSVs. Recover the 31 shared funding
gaps (#54), implement a frozen lake reader/export and specify one distinct
mechanism or horizon before new fits. Six missing ETH funding hours are available
in the older archive, so exchange-wide absence is unproven. See
[the lake runbook](context/lighter_data_lake.md),
[current readiness](experiment-readiness.md) and
[the actual study report](https://github.com/pranay5255/ETHPredict/blob/6f419cc/development-study-results.md).
The AFML reading references below remain a strategy/research guide.

## What the current evidence actually says

The frozen test had 103 trades: -7.264 gross PnL, 128.910 costs and -136.174 net
PnL on 1,000 notional per trade. That is approximately -0.71 bps gross, 12.52 bps
cost and -13.22 bps net per trade. Even eliminating all simulated costs would
leave this particular sample slightly negative. Actual fee/fill measurements may
change economic estimates, but lowering assumptions until a backtest is green
would not establish a viable strategy.

The observed net average winner was 1.829, average loser -3.242, and win rate
37.86%. Holding those net payoff magnitudes fixed, the break-even win rate is
3.242 / (1.829 + 3.242) = 63.93%. This is a conditional arithmetic illustration,
not a forecast, a universal hit-rate requirement or an instruction to target
64% accuracy. Changing exits or selection also changes payoffs.

PSR/DSR are not probabilities that the next month will be profitable. PBO 18.2%
is not an 81.8% chance of making money: it concerns relative ranking under a
particular selection procedure, and our recorded PBO uses PnL rankings while the
actual winner used a classification composite. Those quantities also rely on
sampling and dependence assumptions.

### An urgent forecast-validity finding

Read-only recomputation on 3,279 saved out-of-fold validation forecasts found:

- Selected LSTM-8, next hour: median absolute predicted log return 835.00 bps;
  median absolute realized log return 22.74 bps; MSE 356.18 times a zero-return
  predictor; return rank correlation approximately 0.0172.
- Linear baseline, next hour: median absolute predicted return 20.81 bps;
  MSE 1.51 times the zero-return predictor; return rank correlation about 0.0917.
- Selected LSTM-8, next five minutes: median absolute predicted return 1,659.07
  bps versus 6.41 bps realized. These magnitudes make the predicted-edge filter
  unreliable; a nominal edge threshold cannot fix uncalibrated forecasts.

This is direct evidence of invalid forecast magnitudes for the present
magnitude-based candidate mapping. It does not establish a particular root
cause. The trainer standardizes input features but fits raw small log returns
jointly with a direction loss; the six-epoch budget and initial output scales
make convergence/loss balance plausible suspects. Verify these rather than
assuming a target-unit bug or that more epochs will produce alpha.

Diagnostic JSON:
`artifacts/first_bounded_16_17_20/analysis/forecast_sanity_20261017.json`.
All 16 protected original artifact hashes were checked unchanged after analysis.

## Read AFML in this order, with a concrete output

References were checked in the local `Advances in Financial Machine Learning.pdf`.
Numbers below are printed book pages. In this exact PDF, viewer page = printed
page + 27. Headings and this offset were checked against the content pages;
printed pages 153 and 213 were also rendered and inspected.

1. **Sections 11.4-11.6, pp. 153-157 (PDF 180-184).** Backtesting Is Not a
   Research Tool; A Few General Recommendations; Strategy Selection.
   Output: a written hypothesis, a finite experiment budget and rejection rules.
   Use the failed study to reject its configuration, not to keep mining its
   final-test path. Section 1.2, pp. 4-5 (PDF 31-32), is useful background on
   organizing research as a repeatable process rather than hunting lucky runs.
2. **Sections 15.2-15.4, pp. 211-218 (PDF 238-245).** Symmetric Payouts;
   Asymmetric Payouts; The Probability of Strategy Failure.
   Output: an expectancy sheet with win/loss payoff, hit rate, trading frequency
   and economic break-even conditions. Understand what changes you control and
   which probabilities the market determines. The failure-probability examples
   have modeling assumptions; they are not ready-made posterior probabilities
   for this short dependent crypto sample.
3. **Sections 3.3-3.7, pp. 44-53 (PDF 71-80).** Dynamic thresholds,
   triple-barrier labels, learning side/size and meta-labeling.
   Output: one coherent opportunity/entry/exit/label contract. A meta-model filters
   opportunities from a primary model; it cannot be assumed to manufacture edge
   from arbitrary primary signals. Compare base-only with base-plus-meta.
4. **Sections 8.2-8.4.1, pp. 113-118 (PDF 140-145), and 8.6, pp. 122-126
   (PDF 149-153).** Importance, correlated-feature substitution, permutation,
   single-feature importance and synthetic controls.
   Output: a small predeclared feature-family comparison with noise controls and
   fold variability. Current noise outranking real evaluated inputs is a warning
   against trusting the present ranking; it is not proof that all features lack
   signal. Do not pick features from one favorable importance chart.
5. **Section 9.4, pp. 134-135 (PDF 161-162), then 14.6-14.8, pp. 202-207
   (PDF 229-234).** Scoring, implementation shortfall, efficiency and
   classification metrics.
   Output: separate predictive model quality from after-cost economic acceptance.
   F1/log loss are useful; neither is a profit guarantee. Score probabilities
   against a training-prior baseline, not just raw accuracy, and log return loss
   separately from direction loss.
6. **Sections 7.3-7.4, pp. 104-109 (PDF 131-136), then 12.2-12.5, pp. 161-167
   (PDF 188-194).** Leakage, purging/embargo, walk-forward and CPCV.
   Output: a frozen evaluation map across distinct chronological blocks. The repo
   currently supports grouped scenarios; do not label those as fully stitched
   CPCV backtest paths. Scenarios are not independent observations.
7. **Section 2.5.2, pp. 38-40 (PDF 65-67).** Event-based sampling and CUSUM.
   Output: consider when there is an opportunity worth predicting, instead of
   treating every five-minute bar as equally tradable. CUSUM samples events;
   it does not itself determine a profitable trade direction.

Later, if feature/data evidence warrants it: 19.3-19.6, pp. 282-295 (PDF 309-322),
for spread, impact and microstructural features. Historical book/trade coverage
is required before treating those inputs as available. Read 10.3-10.5,
pp. 142-145 (PDF 169-172), for probability sizing and active bets only after
calibrated positive expectancy exists. Fractional differentiation, ensembles
and TimesFM can be separate studies; they are not the first repair.

## Next task 1: make the forecast numerically credible

Priority: before a broad strategy or hyperparameter search.

- Verify return definitions, timestamps and units end to end: log return,
  fractional return, percent and bps. Test known small examples and prediction
  serialization. Inspect each horizon separately.
- Log raw return MSE/MAE, direction cross-entropy and their weighted gradient/loss
  contributions separately. A declining combined loss can conceal a bad return
  head.
- Test a target standardization per horizon fitted only inside retained training
  rows, persist it with the checkpoint, and invert it on every inference path.
  Early stopping, ensembles, importance refits and benchmark routing must consume
  the same transform. Do not estimate scales from full data or outer validation.
- Verify that the model can fit a tiny controlled dataset; then use a bounded
  longer training schedule with training-only early stopping. Example proposal:
  maximum 50 epochs, patience 5, fixed before the run. More epochs are a technical
  diagnostic, not a strategy-improvement claim.
- Compare zero/mean-return, training-prior direction, ridge/logistic and one small
  corrected LSTM on the same purged chronological development folds. Use a few
  declared seeds, such as 17/29/43, without selecting a lucky seed.
- Save per-fold skill relative to zero MSE, calibration slopes, forecast quantiles,
  directional log loss/Brier versus the training-prior baseline, and rank IC.
  Reject magnitude-based edge estimates while forecasts are grossly mis-scaled.

Start code review in `train_base_model`, `predict_base_model` and
`candidate_signals` in `src/experiments/meta_labeling_mvp.py`, then
`src/evaluation/forecast_diagnostics.py`. Include checkpoint round-trip and
purged refit paths. These references refer to the PR #51 implementation branch.

Passing this task means the training/evaluation machinery is credible. Positive
predictive skill still has to be demonstrated; merely shrinking all predictions
toward zero can improve MSE without creating a monetizable signal.

## Next task 2: formulate and test an economic opportunity

Pick one hypothesis family first. Two candidates for separate investigation are:

- **Continuation:** a range break accompanied by unusually high activity may
  persist while positions adjust. Start with normalized backward returns,
  trailing range position, volume surprise and known volatility.
- **Reversion:** an unusually large move may reverse when temporary pressure
  dissipates. Start with deviation from a trailing price reference, recent
  volatility and a causal trend-state filter. OHLCV variables are only proxies
  for pressure; they do not establish the proposed mechanism.

These are plausible hypotheses to falsify, not discovered edges or book-endorsed
profitable strategies. Write expected behavior, the reason it might persist,
entry information, side rule, exit rule, costs and failure regimes before seeing
new outcome tables. Ensure all features use information actually available at
signal time.

Start with a simple side rule or linear model, and measure its gross conditional
payoffs. Add a logistic meta-filter only as a controlled comparison. Candidate
features should explain when the rule works, rather than duplicate a large set
of price transformations. Keep no-trade, the unfiltered primary rule and the
linear baseline in the report. A constant-prior classifier is a probability
baseline; 50% is not automatically the correct base rate.

For the first study keep one horizon and exit geometry fixed. If one-hour
opportunities cannot pay the measured spread/fees, a separately declared
four-hour comparison is reasonable. Longer holding periods change risk, funding,
labels and purging; larger moves alone do not imply better expected returns.
Do not search horizons, features, thresholds, exits and architectures together.

## Next task 3: validate economic feasibility without chasing PnL

A practical initial protocol, to freeze before running:

1. Mark the old final-test region consumed. Use a declared development partition
   for research and reserve genuinely untouched observations for later confirmation.
   Repeated development CV can also overfit: keep every attempt, failed study and
   follow-up in the research register.
2. Expand historical coverage if available and validated. Aim to include materially
   different trend, volatility and liquidity periods; six to twelve months is a
   useful collection target, not a guarantee or universal minimum. Do not train
   on gaps, mixed fee assumptions or unreliable funding joins.
3. Cap the first signal study at six predeclared configurations and three declared
   seeds. Record actual fits, nested calibration/threshold comparisons and
   feature diagnostics as well as top-level attempts. Review outcomes once the
   batch is complete; do not top up the budget after failures.
4. Use at least three chronological purged evaluation blocks when data permits.
   Compare base-only/base-plus-meta, fold net expectancy, trade count, payoff
   ratio, turnover, exposure, costs and concentration. Fix the trade floor and
   acceptance metric before the study. Abstention is a valid result.
5. Separate model selection from economic acceptance. Retain F1/log-loss and
   calibration diagnostics, but require positive net validation economics and
   robustness across blocks before authorizing another final test. Record the
   classification winner even if it fails the economic gate. The old study's
   winner stays unchanged.
6. Measure the actual account fee tier and executable spread/slippage where
   feasible. Keep the configured 4 bps-per-side scenario as a declared stress
   case if it differs. Test realistic latency/misses and a predeclared cost stress
   such as 2x costs. Strong conclusions require more than one unusually good
   block or a handful of outlier wins. A stress failure identifies dependence
   on favorable execution, not an instruction to lower assumed costs.
7. Estimate uncertainty with a time-block bootstrap or other dependence-aware
   approach on development results. Report intervals for mean net return and
   probability of a specified horizon's loss under explicit resampling assumptions.
   Do not present that as an unconditional live-success probability. Few blocks
   and regime changes may make the answer inconclusive.
8. Only a frozen survivor gets a new untouched confirmation. Then collect forward
   paper-trading evidence with realistic fill accounting and reconcile modeled
   costs against observed execution. Calibrated sizing/exposure caps (#19) are a
   later risk-allocation step; increasing size cannot repair negative expectancy.

Useful first review gates are stable forecast units, improvement over relevant
predictive baselines, plausible features beating noise repeatedly, and positive
net expectancy in most time blocks under measured costs. Requiring survival at
2x costs is a conservative optional gate to declare, not a theorem. A lower
confidence bound on net expectancy above zero is stronger evidence than one
positive mean, but still does not guarantee live returns. If evidence is weak,
reject or defer the hypothesis and investigate a new mechanism without mining
the consumed test partition.

## What to postpone

- Larger LSTMs and very wide hyperparameter grids before forecast sanity passes.
- Variable sizing or leverage as a remedy for current losses.
- Maker-mode profits without queue/fill/adverse-selection evidence.
- The current-state dashboard and TimesFM as prerequisites for learning whether
  the simple signal has edge.
- A fixed win-rate target, Sharpe number or backtest screenshot as the goal.

The first milestone is a small, stable, after-cost conditional edge that survives
an honestly specified validation process. Stop or revise the hypothesis if it
cannot reach that milestone; successful research includes rejecting weak ideas.

Sources: local AFML PDF (2018); saved first-run manifests/OOF forecasts and
`first-experiment-results.md` in the PR #51 worktree. For statistical definitions:
https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf and
https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf .
