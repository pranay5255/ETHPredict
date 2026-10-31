# ETHPredict tasks — autoresearch P3 verified, 2026-10-29

PR #51 and #53 are merged; #16/#17/#20 are closed. Funding repair #54 is
verified on main and closed. P1–P5 now have explicit implementation issues. The
historical backfill completes a data deliverable; it does not close snapshot,
inference or dashboard acceptance criteria. See [project status](PROJECT_STATUS.md)
and [the merge audit](context/pr51_merge_review_20261027.md).

## Frozen autoresearch harness — sequential phases

Build the environment, grader and loop; do not tune a strategy to pass. Each phase
gets its own `autoresearch/p<N>-<slug>` branch and PR, with acceptance evidence
before the next phase. See [design and open decisions](context/autoresearch_design.md).

- [x] P0 housekeeping/design merged in [#60](https://github.com/pranay5255/ETHPredict/pull/60): fresh 317-test baseline, 25-directory/425-hash audit, #54 closed and P1–P5 issues created.
- [x] [#55 P1](https://github.com/pranay5255/ETHPredict/issues/55) Merged in [#61](https://github.com/pranay5255/ETHPredict/pull/61): standalone lake loader, provenance and availability contract; 26,496-row exact CSV parity and 355-test full suite.
- [x] [#56 P2](https://github.com/pranay5255/ETHPredict/issues/56) Merged in [#62](https://github.com/pranay5255/ETHPredict/pull/62): human moved 247 files, 259 real OS denials, code-only agent sandbox with real isolation probes; 428 passed, one historical pre-sealing CSV test skipped. See [runbook](context/grader_tier_sealing.md).
- [x] [#57 P3](https://github.com/pranay5255/ETHPredict/issues/57) Acceptance verified in [PR #63](https://github.com/pranay5255/ETHPredict/pull/63): frozen grader/replay/CLI, trusted feature checks, baseline caches, reports, artifacts and all ten red teams. Full suite: 583 passed, one intentional historical-parity skip. All six full dev baselines and pre-gate calibration are complete; `grader-v1` is pushed at `58228cb`. A separate pinned-checkout production grade passed every integrity check and correctly rejected flat at zero. See [baselines](context/grader_v1_baselines.md) and [acceptance evidence](context/grader_p3_integration_verification.md).
- [ ] [#58 P4](https://github.com/pranay5255/ETHPredict/issues/58) In progress on `autoresearch/p4-loop`: private broker, resumable client, program, protected journals/exports and reviewed runtime provisioner. Focused checks: 49 passed, one optional Docker skip. Human installation and the actual three-attempt private dry run remain; see [deployment runbook](context/autoresearch_private_service.md).
- [ ] [#59 P5](https://github.com/pranay5255/ETHPredict/issues/59) Untuned frozen-head/full-finetune TimesFM seeds with Tier 0 and measured resource reports; GRPO optional.

#30–#35 and #37–#39 are labeled `autoresearch-agenda`. #22/#36/#40 remain
open; grader v1 will take over their parity/cost-stress portions. #19 sizing,
#41 promotion and #43–#45 are outside this construction task. P2 sealing is verified;
P3 strategy-process isolation is implemented and tested with public/synthetic
inputs. The P4 private grading service is not provisioned, and hidden evaluation
is not enabled. No strategy edge is claimed.

## Open implementation and data issues

- [x] [#54](https://github.com/pranay5255/ETHPredict/issues/54) Funding fix merged at c365a81, independently reverified and closed. All 155 payments recovered; zero internal funding gaps. See [P0 audit](context/autoresearch_p0_verification.md).
- [ ] [#19](https://github.com/pranay5255/ETHPredict/issues/19) Probability/edge/concurrency sizing with exposure limits.
- [ ] [#22](https://github.com/pranay5255/ETHPredict/issues/22) Forecast benchmark trading parity, including TimesFM identity/failures.
- [ ] [#43](https://github.com/pranay5255/ETHPredict/issues/43) Recurring completed-bar snapshots with provenance and freshness.
- [ ] [#44](https://github.com/pranay5255/ETHPredict/issues/44) Reviewed frozen checkpoint packaging, reload and current inference.
- [ ] [#45](https://github.com/pranay5255/ETHPredict/issues/45) Local read-only research/current-state dashboard.

## Local integration work not yet covered by a dedicated issue

- [x] Collect five markets and five datasets into ignored monthly Parquet.
- [x] Repair schema compatibility and independently load all 25 directories.
- [x] Verify 425 partition hashes, counts, schemas and timestamp identities.
- [ ] Implement explicit lake ingestion or immutable export with market/date/
  resolution selection, partition provenance and CSV/lake parity checks.
- [ ] Freeze common coverage, units and availability times before side-data use.
- [ ] Specify a new opportunity protocol after the rejected continuation study.

## Open controlled experiments

- [ ] [#30](https://github.com/pranay5255/ETHPredict/issues/30) Weighting.
- [ ] [#31](https://github.com/pranay5255/ETHPredict/issues/31) Validation schemes.
- [ ] [#32](https://github.com/pranay5255/ETHPredict/issues/32) Label geometry.
- [ ] [#33](https://github.com/pranay5255/ETHPredict/issues/33) Forecast calibration/candidate mapping.
- [ ] [#34](https://github.com/pranay5255/ETHPredict/issues/34) Training, seed ensembles and baselines.
- [ ] [#35](https://github.com/pranay5255/ETHPredict/issues/35) Meta models/calibration/thresholds.
- [ ] [#36](https://github.com/pranay5255/ETHPredict/issues/36) Benchmark trading comparison (requires #22).
- [ ] [#37](https://github.com/pranay5255/ETHPredict/issues/37) Feature/noise ablations (requires merged #17).
- [ ] [#38](https://github.com/pranay5255/ETHPredict/issues/38) Search/selection comparison (requires merged #16).
- [ ] [#39](https://github.com/pranay5255/ETHPredict/issues/39) Sizing comparison (requires #19).
- [ ] [#40](https://github.com/pranay5255/ETHPredict/issues/40) Execution/cost comparison (requires merged #20 and valid coverage).
- [ ] [#41](https://github.com/pranay5255/ETHPredict/issues/41) Alpha-claim statistics under a frozen research budget.

## Completed foundations

- [x] #8 activity bars/side-data gates; #9 causal feature reproducibility.
- [x] #10 event sampling; #13 labels; #14 weights; #15 exact-span validation.
- [x] #12 training discipline; #11 forecast diagnostics; #18 meta calibration.
- [x] #16 trial-aware search; #17 purged importance; #20 shared cost/execution model (PR #51).
- [x] #21 statistics; #23 Trackio accounting; #24 artifact/test-use controls.
- [x] #25–#29 historical freeze studies (preserve their original code/data scope).

## Branch studies, not promotion

- [x] PR #51: first bounded fixed-notional study; final test used once, net -136.174.
- [x] PR #53: bounded development study; all economic gates failed, zero new tests.
- [x] PR #51 incorporates the already merged #53 branch and reconciles current main.

The lake is not connected to the general training loader. Funding recovery is
merged, verified and closed in [#54](https://github.com/pranay5255/ETHPredict/issues/54);
schema repair is a separate operation.
More markets, lower thresholds and variable sizing are not evidence of profitability.
