# ETHPredict project status — autoresearch P2 verified, 2026-10-28

The five-market mainnet history lake is collected and readable. It is not yet the
training source, all 31 funding gaps per market are recovered on main, and no
strategy has qualified for promotion.

## Code and evidence boundaries

- Main before the #51 merge: `a97f371` (including PR #52). Causal features, label/split
  controls, weighted training, calibration, side-data gates, statistics, Trackio
  accounting and test-use guards are merged. Issues #8–#15, #18, #21, #23 and #24
  are closed; freeze experiments #25–#29 are closed historical studies.
- [PR #51](https://github.com/pranay5255/ETHPredict/pull/51), branch
  `codex/afml-search-importance-execution`, implements search #16, purged importance
  #17 and execution costs #20. Its merge resolves these three implementation issues;
  its original recorded verification was 218 passing tests.
- [PR #53](https://github.com/pranay5255/ETHPredict/pull/53), branch
  `codex/forecast-opportunity-economics`, was merged into the #51 branch. It adds
  train-only return transforms, checkpoint round trips, one bounded opportunity
  study and an economic gate before final-test access. Recorded verification:
  238 passing tests. The #51 merge brings these changes to main too. Combined-code
  verification and closure scope: [merge audit](context/pr51_merge_review_20261027.md).
- This documentation/EDA/backfill commit series records existing local work.
  Its checks are recorded separately in [project verification](context/project_docs_verification_20261017.md).

## Data inventory

ETH/BTC/SOL/HYPE/NEAR now have monthly mainnet Parquet under ignored
`data/lake/lighter/`, starting 2025-06-04 and ending during 2026-10-16.
The pre-sealing audit loaded all 25 directories and verified all 425 hashes.
After human sealing on 2026-10-28, 275 dev partitions remain public and 150
private partitions are owned by ethgrader; 97 recovery files are quarantined.
The agent verified 259 OS permission denials and a dev-only tool sandbox.
Before funding recovery, Parquet totaled 303.10 MB (289.06 MiB). Candles have zero internal gaps and
mark prices have zero internal gaps after 2025-08-25 15:25 UTC.

Funding's 31 missing hours per market were collection boundary errors. The #54
fix recovered all 155 source payments, preserving original bytes and historical
experiment identities. All 25 directories read successfully and manifests match;
zero internal funding gaps remain. See [verification](context/funding_gap_fix_20261027.md).
The fix is merged at c365a81 and #54 is closed with a fresh 317-test P0 audit.
Full coverage, commands, limitations and machine-readable evidence:
[data lake runbook](context/lighter_data_lake.md).

Training remains ETH CSV-based. Historical side-data groups are opt-in and gated;
none is enabled in the default config. P1 adds the standalone `grader.data` lake
reader with hashes, provenance, half-open windows and availability joins; the
existing training loader is unchanged. Exact ETH CSV parity covers 26,496 rows.
A dev-only audit also rejects malformed HYPE October 2025 1m/5m OHLC partitions.
See [contract/evidence](context/grader_lake_contract.md).

## Actual research results

The first bounded PR #51 study selected on validation and used its final test
once: 103 trades, gross PnL -7.264, costs 128.910 and net PnL **-136.174** at
1,000 notional per trade. All four validation configurations lost money. That
result is evidence against that recipe, not a probability of future profit.
See [the PR #51 report](https://github.com/pranay5255/ETHPredict/blob/81ca53c/first-experiment-results.md).

The PR #53 development continuation used three held-out blocks, six configurations
and three seeds (18 signal/seed trials), plus 36 forecast calls. Unfiltered rules
lost 71.941–78.851 net. Meta-filters abstained or took two trades; the +1.361
two-trade case became -1.309 at 2x costs. **Every configuration failed the frozen
economic gate.** No new final test was evaluated, and the prior protected files
and ledger remained unchanged. Repeated deterministic seeds are not independent
trading evidence. See [the PR #53 report](https://github.com/pranay5255/ETHPredict/blob/6f419cc/development-study-results.md).

These studies used frozen CSV inputs, not general lake ingestion. They are
branch-specific evidence; older archived Trackio runs do not certify current code.
The completed lake supplies more research coverage, not a demonstrated edge.

## Remaining work and order

The active construction task is the frozen autoresearch harness. P0 merged in
[#60](https://github.com/pranay5255/ETHPredict/pull/60); P1 supplies data contracts.
P2 (#56) is verified: the human sealed 150 partitions and 97 recovery files;
ethgrader owns the private root (0700), and the real agent received 259
PermissionErrors. A tested dev-only Docker tool worker excludes old CSV/Git/
artifact copies, with fresh code-only history and no external network.
See [sealing runbook](context/grader_tier_sealing.md). P3–P5 (#57–#59), the frozen
grader/environment, driver and untuned seeds, have not started. No grader-v1 tag or runnable
secure loop exists yet. See [design](context/autoresearch_design.md) and TASKS.md.
Historical backfill alone completes no experiment acceptance criterion.

1. Use the combined #51/#53 code and preserve its economic rejection gate.
2. Preserve the merged funding fix (#54) and verified P1 lake contract. Complete
   human sealing and later harness gates before claiming secure lake-backed grading.
3. Specify one new causal hypothesis, fixed budget, held-out development blocks,
   units, coverage and costs. Retire the tested one-hour continuation recipe.
4. #19 probability/concurrency sizing and #22 benchmark trading parity remain
   open. Fixed-notional studies do not need sizing first; size cannot repair a
   negative expectancy. Full benchmark trading comparisons require parity.
5. #43 completed-bar immutable snapshots → #44 reviewed reloadable checkpoints
   and current outputs → #45 read-only current-state dashboard. Offline bounded
   research does not require this current-state product chain.
6. #30–#41 are controlled comparison studies, still open. Parent implementation
   smoke checks and the first bounded runs do not complete those study matrices.

Use [TASKS.md](TASKS.md) for the issue checklist,
[experiment-readiness.md](experiment-readiness.md) for study gates and
[the implementation roadmap](ethpredict-implementation-roadmap.html) for the visual
status. Do not widen grids, reuse a consumed test, add leverage or promote a model
because infrastructure or data coverage improved.
