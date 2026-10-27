# PR #51 merge review — 2026-10-27

Review source: GitHub REST API through `gh api`, including all 21 open issues,
all 11 merged PRs, PR #51's files, reviews, comments, check runs and commit status.
The primary checkout was clean and was left unchanged; integration used an
isolated worktree.

## Staleness and branch relationship

PR #51 is still needed. Main at `a97f3717a4562d303b22f1e2b557ac9de3ac81b2`
does not contain its search, purged importance or shared execution-cost modules.
The reviewed PR head was `95277637aaf84e47af9487fc3399f7009f5241d9`.
PR #53 was merged into the #51 branch, not main, and its return-transform,
checkpoint and economic-gate changes are therefore part of #51's merge scope.
PR #52's sequential-bootstrap and Trackio fixes are already in that branch's
ancestry and remain intact.

GitHub initially reported `mergeable: false`, `mergeable_state: dirty`.
Merging current main into the review checkout conflicted only in
`context/github_issue_queue.md` and `experiment-readiness.md`. Resolution kept
main's newer lake/funding requirements, retained the measured study evidence,
and reconciled issue counts and PR states. No runtime-code conflict occurred.
The active README, status, task and Trackio notes were updated consistently.
Older dated reports and visual roadmap snapshots remain historical records.

## Issues resolved by the merge

- **#16:** seeded grid/shuffled/random/log-uniform plans, declared and attempted
  budgets, validation-only selectors, all-status accounting, selection roles and
  conservative comparison counts. Integration is in the v2 runner; tests cover
  seed reproducibility, domains, missing/no-trade results and validation selection.
- **#17:** separate base/meta held-out permutation importance, trained noise
  controls, single-feature fits and declared family ablations using the exact
  purged training IDs. JSON/Markdown/Parquet reports and Trackio summaries retain
  unstable and noise-like findings. Tests include a known signal/noise fixture
  and exact-population ablation checks.
- **#20:** a shared cost model for filtering, labels and PnL; causal expected
  quotes and separate realized cashflows; declared historical funding,
  spread/proxy and size/volatility/depth slippage; stress, fills and entry delay.
  Tests cover funding timing/sign/units, missing-source rejection, common cashflows,
  static/stress variants, maker haircuts, delays and capped executable notional.

These are implementation closures. Their broader controlled experiments remain
separate work.

## Issues remaining open

- **#54:** local repair of selected study hours does not repair the five-market
  lake's complete 31-hour gap list or verify collection boundary behavior.
- **#19 and #22:** variable/concurrent sizing and complete benchmark trading
  parity are still incomplete. Forecast diagnostics alone do not establish parity.
- **#43–#45:** an offline artifact dashboard and checkpoint round-trip tests do
  not provide recurring current snapshots, explicit checkpoint promotion and
  complete current inference, or the full current-state dashboard contract.
- **#30–#41:** the small bounded studies do not complete their full frozen
  weighting, validation, labeling, mapping, training, calibration, benchmark,
  feature, search, sizing, execution and claim-statistics comparison protocols.

Expected inventory after closing #16/#17/#20: **18 open issues**, comprising six
implementation/data issues and twelve controlled experiment issues. No strategy
is promoted: the initial final test lost 136.174 net, and the subsequent
development study rejected every configuration without another final-test use.

## Combined-code verification

The full Python suite was run on the combined tree with OMP/OpenBLAS/MKL threads
bounded to one: **243 passed, 21 warnings in 115.72s**, including GPU smoke checks.
Warnings concern constant-series statistical fixtures (statsmodels log likelihood
and undefined Spearman correlations).
The existing Node frontend compatibility and validation/test separation checks
passed. `git diff --check` passed. GitHub had no check runs, commit statuses,
reviews or review comments on the original PR head.
