# ETHPredict implementation plan — 2026-10-17

The next work is to connect verified data to a small, economically gated research
protocol. Earlier label/split/training recommendations are now implemented. Main
at review is `d09efad`; PR #51 and draft PR #53 remain unmerged.

## 1. Finish review of the existing implementation

PR #51 supplies #16 seeded/budgeted search, #17 purged importance/noise controls
and #20 a common cost/execution contract. Its first bounded study lost money;
retain that evidence and its single consumed final test. PR #53 adds train-only
return scaling, reloadable checkpoints, one bounded opportunity study and a
fail-closed economic gate. All its actual configurations were rejected.

Review/merge #51 before integrating its stacked #53. Do not reimplement either
batch on main or describe open PR behavior as already merged.

## 2. Make the mainnet lake an explicit research input

Collection: five markets, five datasets, monthly Parquet from June 4, 2025 through
a partial October 16, 2026. All 25 directories load; 425 hashes match; schemas and
timestamp identities pass. [The runbook](context/lighter_data_lake.md) records the
later mark start and exact bounds. Raw partitions stay ignored in Git.

First fix funding collection boundaries and recover the 31 shared hours (#54).
Six missing ETH hours exist in an older CSV, so do not accept the exchange-wide
gap explanation. Normal refresh and schema repair do not repair closed-month gaps.

Then add a market/date/resolution lake reader or frozen CSV export, with manifest
hashes, duplicate/overlap policy and availability-time/coverage checks. Wire it
through `src/data/features_all.py`, dataset construction, config/schema validation
and run manifests. Verify CSV/lake parity over one declared overlap. Never mix
five markets in an ETH frame or silently change prior experiment inputs.

## 3. Define the next study before fitting

Retire the tested one-hour continuation recipe under its declared costs. Specify
one distinct causal mechanism or longer horizon with a small fixed budget and
purged development blocks. Start with ETH; use other markets only under a declared
replication plan and count market selection in the research budget.

Keep fixed notional, identical comparators, train-only preprocessing, complete
attempt accounting and immutable evidence. Require broad enough trade/block
coverage, positive net expectancy, 2x-cost survival and feature credibility before
allowing any new predeclared final-test evaluation. No profitability probability
can be estimated from the present evidence.

## 4. Finish the remaining products in dependency order

- #19: probability/edge/concurrency sizing, capital/exposure caps and clipping
  audits. Retain fixed notional; sizing is not a remedy for negative expectancy.
- #22: identical benchmark downstream routing, cost/policy fingerprints, TimesFM
  model/runtime identity and explicit skips/failures; separate forecast/PnL ranks.
- #43 → #44 → #45: recurring completed-bar snapshots, reviewed frozen inference
  and read-only dashboard. The lake and archive EDA do not complete these.
- #30–#41: individual controlled comparisons after their own freeze prerequisites.

## Verification and ownership of evidence

Check config/schema and data rejection paths, unit/clock alignment, purging at
both model levels, model reload and per-trade cost reconciliation. Run focused
fixtures, then the repository regression suite. Review scientific claims from
persisted manifests, not dashboard aggregates. Current documentation verification
is [recorded separately](context/project_docs_verification_20261017.md).

See [TASKS.md](TASKS.md) for all 21 open issues,
[experiment-readiness.md](experiment-readiness.md) for study gates and
[the visual roadmap](ethpredict-implementation-roadmap.html) for current status.
