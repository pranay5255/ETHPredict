# Experiment readiness — 2026-10-16

The first useful fixed-notional historical batch has run on the RTX 4090.
PR #50 is merged (`main` base `7e43335`). Search controls #16, purged importance
#17, and costs/execution #20 are implemented on
`codex/afml-search-importance-execution`, pending review and merge.
The primary checkout's concurrent EDA/notebook changes were preserved.

GitHub currently has **20 open issues: 8 implementation and 12 experiment
issues**. Merging this batch closes three, leaving **17: 5 implementation and
12 experiments**. There are no additional implementation issues to finish before
starting bounded fixed-notional historical experiments with this branch. A useful
result can reject a strategy; it does not need to show profitable trading.

## First measured result

Four predeclared configurations completed on frozen local data. All four lost
money after the declared costs on validation. The validation-composite winner,
`lstm_8`, received the only final-test evaluation: **103 trades, -136.174 net PnL,
136.785 maximum drawdown**, on 1,000 requested notional per trade. Its validation
net PnL was -300.895 across 268 trades. No model is promoted.

See [first-experiment-results.md](first-experiment-results.md) for interpretation,
[experimental-runs.md](experimental-runs.md) for commands, and
[the verification record](context/afml_16_17_20_verification.md) for tests and the
reporting-only recovery. Complete raw/model artifacts stay local; the compact
[metrics snapshot](context/first_bounded_backtest_20261016.json) is reviewable.
The frozen ledger records one test use; subsequent dashboard work reused saved
artifacts without new fits, predictions, or test evaluations.

## Remaining implementation issues

- [#19 — probability and concurrency sizing](https://github.com/pranay5255/ETHPredict/issues/19).
  Needed for variable sizing, overlapping active-bet exposure and capital
  efficiency comparisons. Fixed-notional studies can proceed first.
- [#22 — benchmark parity](https://github.com/pranay5255/ETHPredict/issues/22).
  Needed for complete identical-policy cross-forecaster trading comparisons,
  including TimesFM. The first batch compares local linear/LSTM configurations.
- [#43 — refreshed snapshots](https://github.com/pranay5255/ETHPredict/issues/43),
  [#44 — reviewed checkpoint packaging/current inference](https://github.com/pranay5255/ETHPredict/issues/44),
  and [#45 — current-state dashboard](https://github.com/pranay5255/ETHPredict/issues/45)
  form a separate product chain. They do not block offline historical runs.
  The experiment dashboard in this batch does not close these product issues.

## Twelve studies still open

- [#30 weighting](https://github.com/pranay5255/ETHPredict/issues/30),
  [#31 validation schemes](https://github.com/pranay5255/ETHPredict/issues/31),
  [#32 label geometry](https://github.com/pranay5255/ETHPredict/issues/32),
  [#33 forecast/mapping](https://github.com/pranay5255/ETHPredict/issues/33),
  [#34 training discipline](https://github.com/pranay5255/ETHPredict/issues/34), and
  [#35 meta calibration](https://github.com/pranay5255/ETHPredict/issues/35): parent
  implementations are merged; review each issue's freeze requirements.
- [#36 benchmark trading](https://github.com/pranay5255/ETHPredict/issues/36)
  waits for #22; [#39 sizing](https://github.com/pranay5255/ETHPredict/issues/39)
  waits for #19 and verified calibration.
- [#37 feature/noise ablations](https://github.com/pranay5255/ETHPredict/issues/37),
  [#38 search-mode/selector comparisons](https://github.com/pranay5255/ETHPredict/issues/38),
  [#40 cost/execution comparisons](https://github.com/pranay5255/ETHPredict/issues/40), and
  [#41 claim statistics](https://github.com/pranay5255/ETHPredict/issues/41): this
  batch supplies implementation prerequisites and initial diagnostics. A small
  four-trial study does not complete those full comparisons or establish alpha.

Start with a declared validation-only follow-up to the weak forecast/feature
signal and noisy importance findings. Freeze a new hypothesis and budget before
training. Keep this study's consumed final-test partition out of further model
selection. Do not increase leverage or sizing to compensate for negative edge.
