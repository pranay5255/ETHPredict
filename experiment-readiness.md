# Experiment readiness — 2026-10-16

Once PR #50 merges, GitHub has **20 open issues: 8 implementation issues and
12 experiment issues**. Every archived Trackio run predates the causal-feature
(PR #48), label/validation/side-data (PR #49), and training/calibration (PR #50)
changes, so none of it is evidence for the current code.

## Latest batch: #12 → #11 → #18 (PR #50)

Merged in PR #50, which closes all three issues.

- [#12](https://github.com/pranay5255/ETHPredict/issues/12): base LSTM and meta MLP
  early stopping reserve purged holdouts *inside training*. The best checkpoint
  is restored, and selected epochs, fit/holdout IDs, coverage checks, unavailable
  reasons, and per-seed histories are recorded. Config supports deterministic
  seed ensembles, uniform/uniqueness/return weights, ordinary and sequential
  bootstrap, and linear/logistic or random-forest forecast heads on latest-bar
  and rolling summaries. Ensemble return/probability means, member predictions,
  dispersion, and metric variance are saved. Existing validation-to-test rank
  reporting uses only the trials actually permitted a final-test evaluation.
- [#11](https://github.com/pranay5255/ETHPredict/issues/11): forecast reports now
  contain reliability buckets, Brier score, log loss, ECE, return calibration,
  rank correlations, and gross/net edge buckets with coverage. Five candidate
  policies operate on saved predictions without retraining. Zero-return and
  momentum comparisons are explicit; TimesFM has configuration/alignment and
  skip/failure records. Forecast artifacts are separate from simulated PnL.
- [#18](https://github.com/pranay5255/ETHPredict/issues/18): constant, logistic,
  tree, and MLP candidates support training-only Platt/isotonic calibration,
  validation model selection, and threshold selection. Calibration and
  model/threshold selection use disjoint purged populations; MLP stopping uses
  another holdout inside the remaining fit population. Coefficients/importances,
  class balance, fallbacks, threshold provenance, precision/recall/F1, coverage,
  simulated trades/PnL, and calibration are recorded. Final-test threshold sweeps
  are rejected.

## Remaining historical-research implementation work

Continue in the established dependency order:

1. [#16 — search and selection](https://github.com/pranay5255/ETHPredict/issues/16).
   Implement seeded shuffled/random/log-uniform search, record the full proposed
   space and executed budget, and add classification/calibration/edge-rank
   selectors. Keep predeclared validation winners and the final-test ledger.
   Existing all-trial accounting from #23 is a foundation, not completion of #16.
2. [#17 — purged feature importance](https://github.com/pranay5255/ETHPredict/issues/17).
   Add held-out permutation/SFI reports, grouped ablations, and synthetic noise
   controls. Report base and meta importance separately. Tree impurity importance
   from this batch is descriptive and does not replace this validation stage.
3. [#20 — costs and execution](https://github.com/pranay5255/ETHPredict/issues/20).
   Share one cost-model contract across candidates, labels, and PnL. Add historical
   funding, spread/slippage, stress multiples, delayed entries, maker/taker/fill
   assumptions, and break-even diagnostics. Activity-bar funding needs actual
   elapsed time. Validated side-data joins exist; sources still need sufficient
   historical coverage for whichever dynamic model is enabled.
4. [#19 — sizing and concurrency](https://github.com/pranay5255/ETHPredict/issues/19).
   Add calibrated-probability/edge sizing, active-bet accounting across holding
   periods, exposure caps, clipping reasons, and capital-efficiency reports.
   Preserve the fixed-notional comparison.
5. [#22 — full forecast/trading parity](https://github.com/pranay5255/ETHPredict/issues/22).
   Finish comparable benchmark routing contracts, complete runtime/model audits,
   and separate forecast-only versus trading rankings. The expanded diagnostics
   and common splits in this batch are prerequisites, not closure of #22.

The **next implementation batch is #16, #17, and #20**. Then #19
and #22 complete the currently listed historical-research implementation queue.

## Current-state product work

[#43](https://github.com/pranay5255/ETHPredict/issues/43) immutable refreshed
feature snapshots → [#44](https://github.com/pranay5255/ETHPredict/issues/44)
complete reloadable frozen checkpoints and current inference →
[#45](https://github.com/pranay5255/ETHPredict/issues/45) local read-only dashboard.
These are prerequisites for current-state inspection, not for offline historical
experiments. Saving model state here does not claim the complete inference
packaging and explicit checkpoint promotion required by #44.

## Twelve experiments still open

Parent implementation readiness is distinct from permission to run a study.
Every study still needs its own frozen data/config/code identity, split, seeds,
budget, policy, comparison table, and immutable final-test-use record.

- [#30](https://github.com/pranay5255/ETHPredict/issues/30) weighting,
  [#31](https://github.com/pranay5255/ETHPredict/issues/31) validation schemes,
  and [#32](https://github.com/pranay5255/ETHPredict/issues/32) label geometry:
  parent implementations are merged. Review the individual freeze prerequisites
  before running. Grouped scenarios are available; they are not stitched CPCV
  backtest paths.
- [#33](https://github.com/pranay5255/ETHPredict/issues/33) forecast/mapping,
  [#34](https://github.com/pranay5255/ETHPredict/issues/34) training discipline,
  and [#35](https://github.com/pranay5255/ETHPredict/issues/35) meta calibration:
  parent implementations are merged (PR #50).
- [#36](https://github.com/pranay5255/ETHPredict/issues/36) benchmark trading
  comparison waits for #22.
- [#37](https://github.com/pranay5255/ETHPredict/issues/37) feature/noise
  ablations waits for #17.
- [#38](https://github.com/pranay5255/ETHPredict/issues/38) search comparisons
  waits for #16.
- [#39](https://github.com/pranay5255/ETHPredict/issues/39) sizing comparisons
  waits for #19 and verified calibration from #18.
- [#40](https://github.com/pranay5255/ETHPredict/issues/40) cost/execution stress
  waits for #20 and coverage-approved dynamic inputs.
- [#41](https://github.com/pranay5255/ETHPredict/issues/41) alpha-claim statistics
  uses merged #21/#23/#24 and #15 grouped scenarios, but still needs #16 budget
  controls and the fully frozen strategy. Report unavailable statistics honestly.

PR #50 ran synthetic fixtures and software checks, not these performance
experiments. Verification details are in `context/afml_12_11_18_verification.md`.
