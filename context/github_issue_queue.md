# GitHub issue queue

Inventory reviewed through the GitHub API on 2026-10-16: **23 open issues**
(11 implementation issues and 12 follow-up experiments). Primary `main` was
fast-forwarded to `858f2f3` after PR #49 merged; local EDA changes were preserved.

## Current implementation progress

#9 is merged in PR #48. #13, #15, and #8 are merged in PR #49. Their verification
record is `context/afml_13_15_8_verification.md`.

The next batch, #12 → #11 → #18, is implemented on
`codex/afml-training-forecast-calibration`, pending review/merge. Keep issues open
until merge. Verification and configuration are described in
`context/afml_12_11_18_verification.md`. No performance experiments were run.

After this batch merges, continue with **#16 → #17 → #20**, followed by #19 and
#22. #43–#45 form a separate current-state product chain. See the repo-root
`experiment-readiness.md` for every remaining issue and its experiment dependency.

Use isolated branches so primary-checkout EDA work remains intact. Read current
issues/comments, finish cross-file implementation and fixture checks, and leave
reviewable PRs. Existing utilities or archived experiments alone do not establish
completion. #21/#23/#24 are merged foundations; verify their contracts as new
search, benchmark, and execution work is integrated.

## Implementation order

- [#9: AFML: Fix feature reproducibility and fold-local preprocessing](https://github.com/pranay5255/ETHPredict/issues/9) — Remove future-data feature dependencies before any new evidence is trusted. First implementation PR.
- [#13: AFML: Audit triple-barrier geometry and meta-label definitions](https://github.com/pranay5255/ETHPredict/issues/13) — Freeze reconstructable label definitions before changing validation or meta models.
- [#15: AFML: Purge and embargo by exact label spans, then add CPCV](https://github.com/pranay5255/ETHPredict/issues/15) — Use actual label spans for purging and embargo; add CPCV before importance and path claims.
- [#8: AFML: Add activity-driven bars and validated side-data joins](https://github.com/pranay5255/ETHPredict/issues/8) — Wire config-selected side data through alignment and per-split coverage gates.
- [#12: AFML: Add early stopping, uniqueness weighting, and ensemble diagnostics](https://github.com/pranay5255/ETHPredict/issues/12) — Add early stopping and ensemble discipline using trustworthy validation.
- [#11: AFML: Expand base forecast calibration and edge-ranking diagnostics](https://github.com/pranay5255/ETHPredict/issues/11) — Add calibration, rank usefulness, and a shared forecast-to-candidate interface.
- [#18: AFML: Calibrate and benchmark meta-labeler models](https://github.com/pranay5255/ETHPredict/issues/18) — Compare and calibrate meta models after labels and validation are stable.
- [#16: AFML: Make hyperparameter search trial-aware and meta-label scored](https://github.com/pranay5255/ETHPredict/issues/16) — Make search modes and validation-only selection metrics explicit under a fixed budget.
- [#17: AFML: Add purged feature-importance and ablation reports](https://github.com/pranay5255/ETHPredict/issues/17) — Run purged importance, grouped ablations, and noise controls after feature and validation fixes.
- [#20: AFML: Upgrade dynamic cost and execution modeling](https://github.com/pranay5255/ETHPredict/issues/20) — Add execution realism after side-data coverage can be enforced.
- [#19: AFML: Add probability and concurrency-aware bet sizing](https://github.com/pranay5255/ETHPredict/issues/19) — Add probability/concurrency sizing after calibration and cost assumptions are explicit.
- [#22: AFML: Deepen forecast benchmark diagnostics and TimesFM parity](https://github.com/pranay5255/ETHPredict/issues/22) — Route benchmark forecasters through comparable diagnostics and trading logic.
- [#43: Collect refreshed Lighter ETH feature snapshots with provenance](https://github.com/pranay5255/ETHPredict/issues/43) — Write immutable completed-bar current snapshots with provenance and stale/failure states.
- [#44: Freeze reviewed v2 checkpoints and emit read-only current model outputs](https://github.com/pranay5255/ETHPredict/issues/44) — Package complete reloadable models and preprocessing; require explicit local checkpoint promotion.
- [#45: Build a local read-only Lighter research and current-state dashboard](https://github.com/pranay5255/ETHPredict/issues/45) — Build the read-only local dashboard on current-snapshot and frozen-checkpoint contracts.

## Follow-up experiments

Run each only after its parent implementation and all prerequisites in the issue
are available. Freeze the complete research specification and selection budget;
keep validation and final-test evidence separate and obey the final-test ledger.
Code tests and smoke fixtures do not establish performance or justify promotion.

- [#30: AFML Experiment: Compare uniform, uniqueness, return, and combined weights](https://github.com/pranay5255/ETHPredict/issues/30)
- [#31: AFML Experiment: Compare fixed-gap, exact-span, purged k-fold, and CPCV validation](https://github.com/pranay5255/ETHPredict/issues/31)
- [#32: AFML Experiment: Compare triple-barrier geometry and success definitions](https://github.com/pranay5255/ETHPredict/issues/32)
- [#33: AFML Experiment: Compare forecast calibration and candidate mapping value](https://github.com/pranay5255/ETHPredict/issues/33)
- [#34: AFML Experiment: Compare early stopping, seed ensembles, weights, and baselines](https://github.com/pranay5255/ETHPredict/issues/34)
- [#35: AFML Experiment: Compare meta-labeler models, calibration, and thresholds](https://github.com/pranay5255/ETHPredict/issues/35)
- [#36: AFML Experiment: Route benchmark forecasters through identical trading logic](https://github.com/pranay5255/ETHPredict/issues/36)
- [#37: AFML Experiment: Validate feature importance with purged ablations and noise controls](https://github.com/pranay5255/ETHPredict/issues/37)
- [#38: AFML Experiment: Compare search modes and selection metrics under fixed trial budgets](https://github.com/pranay5255/ETHPredict/issues/38)
- [#39: AFML Experiment: Compare fixed, probability, edge, and concurrency sizing](https://github.com/pranay5255/ETHPredict/issues/39)
- [#40: AFML Experiment: Stress static, dynamic, maker, taker, and delayed execution costs](https://github.com/pranay5255/ETHPredict/issues/40)
- [#41: AFML Experiment: Evaluate alpha claims with richer stats, DSR, PBO, and path dispersion](https://github.com/pranay5255/ETHPredict/issues/41)

The closed #28 feature comparison predates the causal #9 changes and did not
promote any features. Preserve that report as historical evidence; future
performance comparisons must identify the new feature-code hash.
