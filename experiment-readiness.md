# Experiment readiness — autoresearch P3 in progress, 2026-10-28

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

The five-market snapshot has 25 datasets and 425 verified partitions, covering
June 2025–October 2026. After human sealing, only its 275 dev partitions are
public; 150 private partitions and 97 recovery files are protected by ethgrader. General training ingestion
remains CSV-based. The #54 fix recovered all 31 funding gaps per market from
source; zero internal gaps remain. The fix is merged at c365a81 and #54 is closed. See
[verification](context/funding_gap_fix_20261027.md) and
[the lake runbook](context/lighter_data_lake.md).

## What must precede the next lake-backed study

1. Use the merged research code and record the actual clean training commit.
   Follow [the merge audit](context/pr51_merge_review_20261027.md) for code scope.
2. Use the repaired funding history and verify its frozen source hashes; reject unsupported
   holding periods. Do not zero-fill. Use market-specific UTC bounds and account
   for the later mark-price start and partial final day.
3. Use the P1 `grader.data` loader with explicit market/resolution/window and
   source hashes/provenance. Its 26,496-row ETH CSV parity check passes;
   availability-time and source schema checks fail closed. HYPE October 2025
   1m/5m malformed OHLC is rejected, not repaired or omitted. See
   [P1 contract](context/grader_lake_contract.md). Use the verified P2 physical isolation and isolated tool worker; complete P3
   frozen grading before starting the new harness.
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

P0 is merged (#60); P1 adds the verified loader. P2 (#56) is merged in #62: human sealing completed, all 259 private opens are denied,
and a dev-only isolated tool worker excludes legacy files/history; see the
[runbook](context/grader_tier_sealing.md). P3 (#57) is in progress; P4–P5 (#58–#59) have not started. The secure grader, freeze tag, driver and TimesFM seeds are not
ready. No open-ended loop or holdout evaluation should be started during construction.
#30–#35 have merged parent controls, but still require individual frozen protocols.
#36 needs #22 benchmark trading parity; #37 and #38 now have their #17/#16 controls;
#39 needs #19 sizing and credible calibration; #40 has #20 but still needs coverage;
#41 needs complete budget/selection accounting and a frozen strategy.

#19 is unnecessary for a fixed-notional opportunity test. #43–#45 are unnecessary
for offline historical research, but required for the current-state product.
Do not wait for every issue to close before a narrow valid study; do not confuse
a functioning experiment pipeline with a positive trading edge.
