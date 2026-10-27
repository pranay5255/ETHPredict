# GitHub issue queue — 2026-10-27 merge review

Repository: [pranay5255/ETHPredict](https://github.com/pranay5255/ETHPredict).
After PR #51 closes #16/#17/#20: **18 open = 6 implementation/data + 12 experiments**.
The issue-by-issue checklist is maintained in [TASKS.md](../TASKS.md).

## Next decisions

1. [PR #51](https://github.com/pranay5255/ETHPredict/pull/51) supplies #16 search,
   #17 purged importance and #20 execution costs. Its merge includes
   [PR #53](https://github.com/pranay5255/ETHPredict/pull/53), which was merged
   into the #51 branch, not directly into main: forecast repair, a bounded
   opportunity study and economic rejection before test access. All actual
   strategy configurations failed that gate. See
   [the merge audit](pr51_merge_review_20261027.md).
2. Merge [#54](https://github.com/pranay5255/ETHPredict/issues/54)'s verified fix:
   month/500-hour boundary behavior is reproduced; all 155 source payments have
   been recovered locally. See [evidence](funding_gap_fix_20261027.md).
3. Add an explicit, frozen lake ingestion/export contract before new lake-backed
   research. This work does not yet have its own dedicated GitHub issue.
4. #19 sizing and #22 full benchmark parity remain implementation work. Keep
   fixed notional until an opportunity demonstrates robust positive expectancy.
5. #43 snapshots → #44 frozen inference → #45 current-state dashboard is a separate
   product chain. Historical backfill and the archived EDA frontend do not meet it.

## Closure audit

The #51 merge resolves #16/#17/#20. No other existing open issue is completed
by the backfill or these bounded studies. #8 validated side-data joins is already
closed; #43/#44/#45 require additional behavior and tests.

Completed implementation: #8–#18, #20, #21, #23, #24. Completed historical study
issues: #25–#29. Experiment issues #30–#41 remain open; initial bounded studies
and synthetic verification do not complete their full comparison protocols.

See [the data audit/runbook](lighter_data_lake.md) and
[the readiness gates](../experiment-readiness.md). Counts are dated snapshots;
query `gh api 'repos/pranay5255/ETHPredict/issues?state=open&per_page=100'`
and exclude pull requests before planning another batch.
