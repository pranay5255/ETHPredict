# Experiment readiness — 2026-10-27 merge review

Bounded offline research has already run on the PR #51/#53 branches. It produced
negative or insufficient economic evidence, not a promoted strategy. The #51
merge includes #53, which was merged into the #51 branch. See
[project status](PROJECT_STATUS.md) for code and result boundaries.

## What is ready

Causal features, label geometry, exact-span validation, training discipline,
forecast diagnostics, meta calibration, statistics, mandatory Trackio accounting
and test-use manifests are merged. PR #51 adds search, purged importance and
execution-cost controls (#16/#17/#20). Its included PR #53 adds
train-only return transforms, exact checkpoint reload verification and economics
before test access. Its 238-test result applies to that branch. Current combined
verification is recorded in [the merge audit](context/pr51_merge_review_20261027.md).

The five-market lake is collected and readable: 25 directories, 425 matching
partition hashes, June 2025–October 2026 coverage. General training ingestion
remains CSV-based. Funding has 31 shared gaps per market, including six ETH hours
present in an older archive. See [#54](https://github.com/pranay5255/ETHPredict/issues/54)
and [the lake runbook](context/lighter_data_lake.md).

## What must precede the next lake-backed study

1. Use the merged research code and record the actual clean training commit.
   Follow [the merge audit](context/pr51_merge_review_20261027.md) for code scope.
2. Recover funding gaps or document direct source unavailability; reject unsupported
   holding periods. Do not zero-fill. Use market-specific UTC bounds and account
   for the later mark-price start and partial final day.
3. Implement explicit lake loading or immutable CSV export. Record selected market,
   resolution, raw partition hashes, overlap/dedup policy, units, feature schema,
   availability-time joins and per-split coverage. Prove parity on a fixed overlap.
4. Freeze one new mechanism/horizon and a small comparison budget. The tested
   one-hour continuation recipe is rejected; its weak result does not justify
   larger grids. Count models, rules, markets, thresholds and diagnostic fits.
5. Use purged development blocks, train-only transforms/calibration, fixed notional,
   realistic costs and explicit no-trade outcomes. Keep comparisons on the same
   rows, opportunity set, labels, costs and downstream policy.
6. Demand sufficient trades and blocks, positive aggregate after-cost expectancy,
   stability across blocks, 2x cost survival and credible feature/noise evidence.
   Freeze numeric gates before running. Missing evidence must fail closed.
7. Only a validation-selected qualifying policy may enter a fresh predeclared
   final holdout. The prior final test was consumed once. Do not rerun it or treat
   an expanded lake as resetting the old test-use ledger.

The PR #53 frozen gate required 60 trades overall, ten per block, two positive
blocks out of three, positive aggregate net and 2x-cost net, plus feature evidence.
Every configuration failed. These are that study's gates, not a universal proof
of profitability or automatic approval for a new study.

## Which remaining issues block which experiments

After the #51 merge closes #16/#17/#20, 18 issues remain: 6 implementation/data
and 12 experiments.
#30–#35 have merged parent controls, but still require individual frozen protocols.
#36 needs #22 benchmark trading parity; #37 and #38 now have their #17/#16 controls;
#39 needs #19 sizing and credible calibration; #40 has #20 but still needs coverage;
#41 needs complete budget/selection accounting and a frozen strategy.

#19 is unnecessary for a fixed-notional opportunity test. #43–#45 are unnecessary
for offline historical research, but required for the current-state product.
Do not wait for every issue to close before a narrow valid study; do not confuse
a functioning experiment pipeline with a positive trading edge.
