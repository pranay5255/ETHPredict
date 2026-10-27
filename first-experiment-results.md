# First bounded ETH backtest — 2026-10-16

**Decision: reject this configuration as a profitable trading strategy under the
declared assumptions.** All four configurations lost money after costs on
validation. The selected configuration also lost on the one final-test
assessment. The pipeline now provides actionable diagnostics, but this study
provides no basis for model promotion or larger sizing.

## Frozen protocol

Run: `first_bounded_fixed_notional_v1_20261016T175303Z`.
Training code: clean commit `6aeb87cdfc8ae6d63e5b3178a6d57194d6932e35`.
Device: NVIDIA GeForce RTX 4090, `cuda:0`; this was not a smoke run.

The chronological source window was 2026-05-30 13:20 through 2026-06-27 23:55 UTC:
8,192 five-minute bars, including 2,016 warmup bars. There were 6,164 samples;
4,919 development and 1,233 test rows, with purged boundary rows excluded as
recorded in the split manifests. Two purged outer folds had zero reported
train/validation label-span overlaps. Features were OHLCV, returns, realized
volatility and volume, plus two trained random noise controls. Forecast heads
predicted next-five-minute and next-hour outcomes; trading used next-hour
candidates and a twelve-bar cooldown.

The predeclared seed was 17; four shuffled attempts covered linear, LSTM-8,
LSTM-16 and regularized LSTM-16. Training was bounded at six epochs with internal
purged early stopping. Logistic meta models used training-only Platt calibration
and F1-based threshold selection. The outer validation selector was
`F1@0.5 - 0.5 * ECE + 0.1 * edge_rank`, with a minimum of 30 trades. It was not a
net-profit selector. Requested and capped notional were both 1,000; capital was
100,000. Probability/concurrency sizing and TimesFM parity were excluded.

Costs used configured 4 bps per-side taker fees, a declared high/low spread proxy,
volatility-sensitive slippage, and historical signed funding payments with
explicit percent units and payer direction. One missing hourly payment was
recovered by an exact-timestamp public request during preparation; provenance
and response hash are in the freeze manifest. The archive itself was unchanged.
These assumptions do not reproduce account-specific fees, queueing or live fills.

## Validation outcomes

- **Linear:** 206 trades; net **-239.951**; meta F1@0.5 **0.000**; ECE **0.02395**.
  It was the PnL, calibration and edge-rank winner, and still lost after costs.
- **LSTM-16:** 268 trades; net **-296.218**; meta F1@0.5 **0.00464**.
- **LSTM-8:** 268 trades; net **-300.895**; meta F1@0.5 **0.06850**.
  This was the frozen composite and classification winner.
- **Regularized LSTM-16:** 265 trades; net **-330.838**; meta F1@0.5 **0.000**.
  It was the negative-log-loss winner.

All attempts completed; none were replaced. The linear baseline's stronger PnL
was recorded without changing the predeclared selector or scoring it on test.
The selected LSTM's small positive gross validation PnL (11.670) was overwhelmed
by 312.565 in costs. Its frozen-outcome validation net PnL fell to **-613.815** at
2x costs and **-1,552.574** at 5x costs. These are PnL haircuts, not relabeled and
refitted cost-regime studies.

## Only final-test assessment: LSTM-8

- **103 trades**, **-7.264 gross PnL**, **128.910 total costs**, **-136.174 net PnL**.
- Maximum drawdown **136.785**; win rate **37.864%**; turnover **103,000**.
- Additional break-even slippage **-13.221 bps/trade**: it was already below
  break-even, even before adding further slippage.
- Fees **82.400**, spread **31.744**, slippage **14.736**, funding **0.030**.
- PSR **2.50e-8**, DSR **1.03e-11**; annualized Sharpe **-42.230** describes a
  short historical sample, not a forward return projection.

The conservative disclosed comparison count was **108**: four top-level
attempts, 48 nested threshold comparisons and 56 SFI/group-ablation fits. These
are dependent comparisons; DSR is exploratory, with strategy-Sharpe variance
estimated from only four configurations. PBO was **0.1821** from validation PnL
rankings over four strategies, 16 CSCV slices and 6,113 combinations. It does not
directly estimate overfitting of the classification-composite selector. The
result is explicitly graded `exploratory_backtest`, with no alpha claim.

## What to investigate next

The trained-noise diagnostics flagged `diagnostic_noise_0` above the evaluated
real base inputs for the selected LSTM. Both noise controls were flagged for the
other LSTMs; none for linear. This is evidence against trusting the present
feature/model ranking. It does not establish that a particular feature causes
returns. The dashboard retains fold variability, base/meta permutation scores,
single-feature refits and family-removal reports.

A sensible next study is a small, predeclared validation-only comparison of
forecast scaling/convergence and feature families against the linear baseline.
The six-epoch bound and one seed leave training adequacy unresolved. Validate
forecast magnitudes and held-out edge calibration before a larger search.
Require positive net validation economics as well as acceptable classification
quality before allocating another untouched final-test partition. Do not tune
against this study's now-consumed test sample or switch its winner retroactively.

## Artifacts and reporting recovery

Local root: `artifacts/first_bounded_16_17_20/` in the primary checkout.
Dashboard: `runs/first_bounded_fixed_notional_v1_20261016T175303Z/report/dashboard.html`.
Serve that root on port 8767 as described in [experimental-runs.md](experimental-runs.md).
A standalone copy is `~/.agent/diagrams/ethpredict-first-experiment.html`.
The compact [metrics and provenance snapshot](context/first_bounded_backtest_20261016.json)
is tracked; raw data, checkpoints and full dashboards remain local.

The original process finished all training/test/statistics, then failed during
Trackio readback because its CLI was outside the shell PATH. Reporting was
recovered from saved outcomes after adding a virtualenv CLI fallback. The
legacy presence-only evidence grade also incorrectly called available statistics
alpha-grade; the corrected assessment evaluates numeric outcomes and marks this
first study exploratory. The original manifest is preserved. Audited model,
prediction, candidate, test-trade and ledger hashes are unchanged; reporting
added **zero** fits, predictions or test evaluations. All four trial runs plus
the summary and accounting runs were read back from local Trackio.
