# GitHub issue queue — 2026-10-17

Repository: [pranay5255/ETHPredict](https://github.com/pranay5255/ETHPredict).
Snapshot after opening #54: **21 open = 9 implementation/data + 12 experiments**.
The issue-by-issue checklist is maintained in [TASKS.md](../TASKS.md).

## Next decisions

1. Review [PR #51](https://github.com/pranay5255/ETHPredict/pull/51): #16 search,
   #17 purged importance and #20 execution costs are implemented there, but the
   PR is unmerged and the three issues remain open.
2. Review [draft PR #53](https://github.com/pranay5255/ETHPredict/pull/53), stacked
   on PR #51: forecast repair, a bounded opportunity study and economic rejection
   before test access. All actual strategy configurations failed that gate.
3. Fix [#54](https://github.com/pranay5255/ETHPredict/issues/54): investigate
   month/500-hour funding boundaries and recover historical gaps. Six missing
   ETH hours exist in the older archive; exchange-wide absence is unproven.
4. Add an explicit, frozen lake ingestion/export contract before new lake-backed
   research. This work does not yet have its own dedicated GitHub issue.
5. #19 sizing and #22 full benchmark parity remain implementation work. Keep
   fixed notional until an opportunity demonstrates robust positive expectancy.
6. #43 snapshots → #44 frozen inference → #45 current-state dashboard is a separate
   product chain. Historical backfill and the archived EDA frontend do not meet it.

## Closure audit

No existing open issue is completed by the backfill alone. #8 validated side-data
joins is already closed. #16/#17/#20 wait for their PR merge; #43/#44/#45 require
additional behavior and tests. No issue was closed merely to reduce the count.

Completed implementation: #8–#15, #18, #21, #23, #24. Completed historical study
issues: #25–#29. Experiment issues #30–#41 remain open; initial bounded studies
and synthetic verification do not complete their full comparison protocols.

See [the data audit/runbook](lighter_data_lake.md) and
[the readiness gates](../experiment-readiness.md). Counts are dated snapshots;
query `gh issue list --state open --limit 100` before planning another batch.
