# Development-only forecast and continuation study

This implements the three tasks in the attached instructions: repair forecast
training/verification, investigate one explicit opportunity, and require economic
acceptance before a future final test. It builds on open PR #51 and the merged
PR #52 fixes in an isolated branch. The first study's test remains consumed.

## Forecast repair

`training.target_standardization: standard` estimates a per-horizon mean/std on
retained training rows after the internal purge. LSTM outputs are converted to
natural log-return fractions in the model itself. Checkpoints persist input and
target transforms; the reload helper reconstructs models/ensembles and includes
identity transforms for historical LSTM states. Training records raw return MSE,
standardized return objective, direction cross-entropy, combined objective and
actual loss weights for training and internal validation, separately.

The new recipe permits at most 50 epochs, patience 5, and gradient clipping.
Zero and training-mean return baselines use retained-training direction priors.
Zero, mean, ridge/logistic and small LSTM models share the exact outer folds and
three declared seeds. Every validation checkpoint is reloaded and compared for
bit-exact predicted returns/probabilities. Tests check unit/timestamp alignment,
train-only transforms, learnability of a controlled small-return signal and
checkpoint round trips. Numerical validity is not a claim of predictive skill.

## One opportunity hypothesis

A close beyond the preceding 24-hour high/low range, accompanied by unusual
volume, might continue. This is a hypothesis to falsify, not an established edge.
The most recent input bar is completed before the entry bar. Range and volume
statistics exclude that input bar from their estimation window. The one-hour
horizon, asymmetric barriers, 1,000 fixed notional, cooldown and cost model stay
fixed. Six configurations cover volume-z minima 0/0.5/1, each with and without a
logistic meta-filter. Seeds 17/29/43 are disclosed repetitions, not independent
market evidence. The rule-only side generator is exogenous; no fitted primary
forecast can leak its training outcomes into the meta stage.

The rule does not pretend its breakout distance is an expected return or attach
a fabricated probability. Expected-edge ranking is explicitly not estimated;
zero is a neutral scheduling score. The meta layer sees known market/rule inputs
and costs, with two trained noise controls. It uses disjoint purged calibration
rows inside training and a fixed threshold. Held-out permutation importance and
fold variability qualify feature credibility. No threshold sweep is performed.

## Economic acceptance and test isolation

The development-only runner imports no final-test scorer. It requires evaluation
to be disabled, at least three chronological purged folds and source data ending
before 2026-05-30, which excludes the entire earlier experiment window. An unused
reserve is also excluded by the standard split builder; it is not evaluated or
claimed to be a newly certified untouched final-test sample.

A configuration must pass in every declared seed: positive total net PnL, at
least two positive blocks of three, 60 total trades, at least ten per block,
positive net PnL at 2x costs, and stable positive rule-feature importance above
trained noise. Missing evidence fails the gate. The general pipeline now also
supports a research economic gate before its test scorer/ledger, preserving the
classification winner even when economics rejects it. The scorer itself checks
again before any ledger write; the legacy benchmark rejects economic-gated
final-test comparisons until it supports validation-selected acceptance. Enabling gates cannot
retroactively change the old study's winner or its consumed test outcome.

Costs retain the conservative configured fees, historical signed funding,
declared spread proxy and volatility slippage. The 2x/5x reports freeze candidates,
probabilities, exits and policy and are PnL-only haircuts; they do not claim
regenerated supervised labels. Maker queueing or measured live fees/fills are not
assumed. A successful gate would authorize review, not deployment or automatic
final-test access.

## Run and inspect

The recipe is `configs/development_continuation.yml`. Preparation copies the
chronological tail before the cutoff, verifies unique complete five-minute bars,
repairs exact missing funding hours from declared local parquet archives without
network fallback, and hashes frozen inputs/config. Existing study directories
cannot be overwritten. Real runs require clean committed source. All fits,
failures, no-trade outcomes and comparison budgets remain in artifacts. Local
Trackio receipts cover each forecast fit, each signal/seed trial and the summary.

```bash
uv run python -m src.experiments.local_batch \
  --template configs/development_continuation.yml \
  --source-data /path/to/primary/checkout/data \
  --output-root /path/to/new/frozen-study --prepare-only

OMP_NUM_THREADS=1 uv run python -m src.experiments.development_study \
  --prepared-config /path/to/new/frozen-study/frozen_config.yml \
  --output /path/to/new/frozen-study/results --device cuda:0
```

For this first actual run, the preparation template also enrolled the prior
study's 16 protected model/prediction/candidate/test-trade/ledger files plus its
frozen config/manifest in `development_study.protected_artifacts`. Their hashes
are recorded in `protocol.json` and checked unchanged after the study. A source
end cutoff additionally keeps their data out of new development selection.

Inspect `protocol.json`, `splits.json`, `attempts.json`, per-fit checkpoints,
predictions and loss histories, per-trial candidates/trades/purge audits,
`report.json` and `report.md`. Displayed results are development evidence only.
The comparison budget distinguishes six configurations, eighteen seed trials,
36 forecast fit calls, meta fits/calibrators and permutation evaluations. These
are not a claim of independent trials or a complete lifetime research count.

If logging fails after outcomes are saved, log the saved report without any
additional model work:

```bash
uv run python -m src.experiments.development_study \
  --log-report /path/to/new/frozen-study/results/report.json
```

Existing locally verified receipts are reconciled. Never rerun training just to
recover a report. Read the final result/verification record before deciding what
new hypothesis, if any, deserves another frozen development study.

The completed study and its rejection decision are recorded in
[development-study-results.md](development-study-results.md).
