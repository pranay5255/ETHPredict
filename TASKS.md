# ETHPredict tasks — 2026-10-17

GitHub has **21 open issues: 9 implementation/data and 12 experiments**. The
historical backfill completes a data deliverable; it does not close snapshot,
inference or dashboard acceptance criteria. No existing open issue was closed
in this update. See [project status](PROJECT_STATUS.md).

## Open implementation and data issues

- [ ] [#16](https://github.com/pranay5255/ETHPredict/issues/16) Trial-aware search — implemented in open PR #51; pending merge.
- [ ] [#17](https://github.com/pranay5255/ETHPredict/issues/17) Purged feature importance — implemented in open PR #51; pending merge.
- [ ] [#20](https://github.com/pranay5255/ETHPredict/issues/20) Dynamic cost/execution interface — implemented in open PR #51; pending merge.
- [ ] [#54](https://github.com/pranay5255/ETHPredict/issues/54) Recover the 31 shared funding collection gaps and verify boundaries.
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
- [x] #21 statistics; #23 Trackio accounting; #24 artifact/test-use controls.
- [x] #25–#29 historical freeze studies (preserve their original code/data scope).

## Branch studies, not promotion

- [x] PR #51: first bounded fixed-notional study; final test used once, net -136.174.
- [x] PR #53: bounded development study; all economic gates failed, zero new tests.
- [ ] Merge/review their code before treating these capabilities as merged main.

The lake is not connected to the general training loader. Funding recovery is
[#54](https://github.com/pranay5255/ETHPredict/issues/54), not schema repair.
More markets, lower thresholds and variable sizing are not evidence of profitability.
