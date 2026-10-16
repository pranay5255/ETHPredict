# GitHub issue queue

Inventory reviewed on 2026-10-16: **20 open issues** (8 implementation issues and
12 follow-up experiments) after PR #50 merged. This is a proposed dependency
order, not a claim that the backlog is complete.

## Current implementation progress

Merged foundations: #21, #23, and #24 (PR #47); #9 (PR #48); #13, #15, and #8
(PR #49); #12, #11, and #18 (PR #50). Verification records are
`context/afml_13_15_8_verification.md` and `context/afml_12_11_18_verification.md`.
No performance experiments were run on the merged code; every archived Trackio
run predates PR #48–#50.

Next: **#16 → #17 → #20**, followed by #19 and #22. #43–#45 form a separate
current-state product chain. `experiment-readiness.md` lists every remaining
issue and its experiment dependency.

Work one implementation issue per isolated branch and PR: read current issues
and comments, finish cross-file implementation and fixture checks, and leave a
reviewable PR. Leave the issue open until its fix is merged. Existing utilities
or archived experiments alone do not establish completion; verify the merged
contracts as new search, benchmark, and execution work is integrated.

## Implementation order

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
