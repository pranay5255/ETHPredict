# Forecast repair and continuation study - 2026-10-17

**Decision: no configuration qualifies for a final test.** Forecast magnitudes
are numerically plausible under the new training contract, but consistent skill
against a zero-return predictor was not demonstrated. The continuation rule lost
after costs; its meta-filter mostly abstained. A two-trade positive outcome was
correctly rejected by the predeclared coverage and cost gates.

## What ran

Clean training commit: `a33e6c45cf80b1687724bc04983bf4c567120781`, based on open
PR #51 plus merged `main`/PR #52 fixes. Device: `cuda:0`, NVIDIA RTX 4090.
Source: 16,384 completed five-minute bars, 2026-04-03 02:40 through 2026-05-29
23:55 UTC; 2,016 warmup bars. This excludes the entire earlier study window,
which began 2026-05-30. The new split retained an unused reserve; it was never
evaluated and is not claimed to be a newly certified untouched confirmation set.

Three outer chronological blocks each had 2,868 validation rows. Retained outer
training rows were 2,857 / 5,725 / 8,593, with exact-span purging. Each LSTM also
reserved a purged internal holdout for stopping and target-transform fitting.
Forecasts and opportunities used the same outer development split.

The first preparation attempt failed with HTTP 429 before training. Its failure
record is retained. Three exact missing funding hours were available in the
local backfill lake and copied with source hashes/provenance into the successful
freeze. No further network requests, source-archive edits, date-window changes,
trial-budget extensions or outcome-based replacement trials were used.

## Forecast checks

Zero/mean-return, ridge/logistic and one small LSTM ran on identical outer folds,
with seeds 17/29/43. There were 36 forecast fit calls, including train-only
constant baselines and declared seed repetitions. The LSTM permitted 50 epochs
maximum with internal early stopping and patience 5. It emitted natural log-return
fractions from train-only standardized targets, with checkpointed input/target
transforms. Separate raw-return MSE, standardized objective and direction loss
histories are saved. All 36 validation checkpoints reloaded with bit-exact
predictions/probabilities.

Across the nine seed/fold cases per model, median MSE relative to predicting zero:

- **Zero:** 1.0000, with zero predicted return by definition.
- **Training mean:** 1.0019; median absolute prediction 0.863 bps.
- **Ridge/logistic:** 1.2225; median absolute prediction 10.054 bps.
- **LSTM:** 1.0142; median absolute prediction 4.649 bps.

Only two of nine LSTM seed/fold cases beat zero MSE; the range was 0.9796-1.0696.
These cases are dependent, not nine independent demonstrations. Controlled
synthetic tests verify that the model can learn a known small-return signal.
Actual fit means/scales were independently checked against retained training
returns; timestamps/target fractions were checked against raw candle prices.

The earlier 835 bps forecast magnitudes and 356x zero-MSE error came from a
*different* window and recipe. This is not a controlled estimate of the causal
before/after effect of scaling. The defensible finding is that this implementation
passes unit/transform/convergence sanity checks while these new development
results still do not show a dependable predictive advantage.

## One continuation hypothesis

The side rule used the previous completed close breaking the preceding 24-hour
high/low, conditional on volume surprise. The range/volume reference excluded the
signal bar. Volume-z minima 0 / 0.5 / 1, with and without a logistic meta-filter,
made six configurations; each was evaluated for the three declared seeds.
One-hour horizon, barriers, 1,000 fixed notional, cooldown and cost model stayed
fixed. The rule never pretended breakout distance was an expected return or
assigned an invented probability. Known rule/market-state inputs and trained
noise controls were used by the logistic overlay; calibration was purged inside
training and the decision threshold stayed fixed.

All seeds produced identical trading outcomes for these deterministic rule and
logistic fits. Do not multiply their trade counts or treat those repeats as
independent performance evidence.

- **Volume z >= 0, unfiltered:** 55 trades, **-71.941** net; all three blocks lost.
- **Volume z >= 0.5, unfiltered:** 54 trades, **-74.990** net; all three blocks lost.
- **Volume z >= 1, unfiltered:** 53 trades, **-78.851** net; all three blocks lost.
- **Volume z >= 0 or 0.5, meta-filtered:** zero trades. Abstention was retained.
- **Volume z >= 1, meta-filtered:** two trades, **+1.361** net in only one block;
  **-1.309 at 2x costs**, **-9.319 at 5x costs**.

The unfiltered z >= 0 rule deteriorated to -142.385 at 2x costs and -353.717 at
5x costs. Stress reports are frozen-outcome PnL haircuts, not relabeled/refitted
cost-regime studies. Configured fees and spread proxies remain simulations;
maker queueing or actual account-specific fills are not assumed.

## Why the gate rejected every configuration

The frozen acceptance required three blocks, positive total net PnL, at least
two positive blocks, at least 60 trades overall and ten per block, positive net
PnL at 2x costs, and stable rule-feature importance above trained noise. Missing
evidence failed. The unfiltered rules lost across every block; the meta-filtered
rules lacked coverage, and the two-trade positive case failed cost stress and
feature credibility. No thresholds were lowered or budgets increased afterward.

All eighteen signal/seed trials and 36 forecast calls are in `attempts.json`,
including six no-trade trials. Counts also disclose 27 logistic fit calls,
27 calibration fits and 972 permutation evaluations. These are within-study
counts, not a complete lifetime multiple-testing adjustment.

## Verification and next decision

The completion audit verified all frozen hashes, chronological spans, unit/clock
alignment, actual train-only target statistics, all checkpoint round trips and
all gates recomputed from saved validation evidence. All **55** Trackio records
were read back locally. The prior study's **18** enrolled protected files are
unchanged, including all sixteen outcome/model/ledger files plus its frozen
config/manifest. The old ledger remains at one use. This study added **zero**
final-test evaluations and created no new test ledger.

Final test-entry guards were hardened after the study: economic rejection also
blocks direct scorer calls before any ledger write, and the legacy all-model
benchmark cannot bypass the economic requirement. These guard changes do not
change the scientific outputs or authorize another run.

The next research decision is to retire this particular one-hour continuation
recipe under the declared costs. A different causal feature set, a separately
specified longer horizon or a different opportunity mechanism can be investigated
in a new development protocol. The result does not justify larger search grids,
variable sizing, leverage or final-test reuse.

See [development-study.md](development-study.md) for implementation and commands,
and [the compact result/audit snapshot](context/development_continuation_20261017.json)
for review. Full models/predictions/trades/losses remain local at
`artifacts/development_continuation_20261017_local/development_v1/` in the primary
checkout; `report.md`, `report.json`, `completion_audit.json` and per-trial
manifests provide the detailed audit trail.

Final software verification: **238 tests passed** (21 synthetic-series warnings).
Logging-only replay also preserved all **112** current scientific output hashes.
