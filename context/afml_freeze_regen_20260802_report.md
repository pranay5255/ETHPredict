# AFML Freeze Regeneration Report - 2026-08-02

Generated from current local GPU runs for GitHub issues #25-#29.

Artifact root:

```text
artifacts/afml_freeze_regen_20260802
```

Environment observed during regeneration:

- Device: `cuda:0`
- GPU: `NVIDIA GeForce RTX 4090`
- Torch: `2.11.0+cu128`
- Git commit recorded by manifests: `16c9c159f534eb46dc1bf31ef17b28d113eb44ee`
- Git dirty state recorded by regenerated manifests: `false`

## Executive Result

- Issue #25 passed as a reproducibility/accounting gate: identical baseline reruns matched config hash, raw data hashes, split hashes, trial count, trial IDs, and metrics within tolerance. The controlled changed-config run failed comparison as expected while raw hashes and split hashes stayed stable.
- Issue #26 passed as a trial-accounting gate: completed, failed, skipped, no-trade, low-trade, and trade-qualified cases were all represented in the local artifacts and Trackio runs.
- Issues #27-#29 completed as negative/no-promotion experiments: bar-clock, feature-family, fracdiff, and event-sampling variants regenerated, but none produced validation or test trades under the frozen `0.55` meta-label threshold.
- These runs validate instrumentation and research controls. They do not validate tradable alpha.

## Issue #25: Reproducibility Gate

| Run | Run id | Final-test reuse count | Validation trades | Test trades |
| --- | --- | ---: | ---: | ---: |
| Baseline A | `25_repro_baseline_a_20260802_20260802T144917Z` | 1 | 0.0 | 0.0 |
| Baseline B | `25_repro_baseline_a_20260802_20260802T145148Z` | 2 | 0.0 | 0.0 |
| Changed config | `25_repro_config_changed_20260802_20260802T145405Z` | 1 | 0.0 | 0.0 |

Baseline comparison command:

```bash
uv run python -m src.experiments.reproducibility \
  artifacts/afml_freeze_regen_20260802/25_repro_baseline_a_20260802_20260802T144917Z \
  artifacts/afml_freeze_regen_20260802/25_repro_baseline_a_20260802_20260802T145148Z \
  --metric-tolerance 0.01
```

Baseline comparison result:

```json
{
  "status": "pass",
  "checks": {
    "config_hash_match": true,
    "raw_hashes_match": true,
    "split_hashes_match": true,
    "trial_count_match": true,
    "trial_ids_match": true,
    "metrics_within_tolerance": true
  }
}
```

Changed-config comparison result:

```json
{
  "status": "fail",
  "checks": {
    "config_hash_match": false,
    "raw_hashes_match": true,
    "split_hashes_match": true,
    "trial_count_match": true,
    "trial_ids_match": true,
    "metrics_within_tolerance": false
  }
}
```

Observed changed metric fields were the configured `edge_threshold_bps` and `effective_edge_threshold_bps`, changing from `5.0` to `7.0`.

Decision: close #25 as completed. Keep the caveat that the gate proves manifest/test-reuse instrumentation, not alpha quality.

## Issue #26: Trial Accounting

Run id:

```text
26_accounting_matrix_20260802_20260802T145622Z
```

Counts:

```json
{
  "completed": 3,
  "failed": 1,
  "skipped": 1,
  "no_trade": 1,
  "low_trade": 2,
  "trade_qualified": 1
}
```

Selection summary:

- Raw best trial: `no_trade`
- Raw best selection status: `abstention`
- Trade-qualified best trial: `normal_trade`
- Classification best trial: `no_trade`
- Calibration best trial: `normal_trade`
- Qualified trial count: `1`

Decision: close #26 as completed. The important result is that failure/no-trade/low-trade cases are visible and cannot silently masquerade as a clean winning run.

## Issue #27: Bar Clock

| Variant | Run id | Rows | Coverage | Validation trades | Test trades |
| --- | --- | ---: | ---: | ---: | ---: |
| Time bars | `27_bars_time_20260802_20260802T150137Z` | 105074 | 1.0000 | 0.0 | 0.0 |
| Volume bars | `27_bars_volume_20260802_20260802T150334Z` | 10036 | 0.0955 | 0.0 | 0.0 |
| Dollar bars | `27_bars_dollar_20260802_20260802T150344Z` | 10222 | 0.0973 | 0.0 | 0.0 |

Decision: close #27 as experiment complete with no promotion. Keep 5-minute time bars as the baseline for now. Volume and dollar bars change the data distribution and reduce sample count substantially, but they did not produce tradable validation evidence under the frozen policy.

## Issue #28: Feature Ablation

| Variant | Run id | Inputs | Feature families | Fracdiff mode | Scaler fit scope | Validation trades | Test trades |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| OHLCV only | `28_features_ohlcv_20260802_20260802T150406Z` | 6 | 1 | `none` | `train_indices_only` | 0.0 | 0.0 |
| Full no fracdiff | `28_features_no_fracdiff_20260802_20260802T150536Z` | 23 | 8 | `none` | `train_indices_only` | 0.0 | 0.0 |
| Full fixed fracdiff | `28_features_fixed_fracdiff_20260802_20260802T150710Z` | 24 | 9 | `fixed_width` | `train_indices_only` | 0.0 | 0.0 |

Fixed-width fracdiff diagnostics:

- ADF p-value: `0.8436306474491656`
- Memory correlation: `0.999444551874062`
- NaN rows: `9`

Decision: close #28 as experiment complete with no feature promotion. Fold-local scaling is working, but the tested feature variants did not overcome the no-trade policy behavior.

## Issue #29: Event Sampling

| Variant | Run id | Events | Event rate | Average uniqueness | Effective N | Validation trades | Test trades |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | `29_sampling_dense_20260802_20260802T150918Z` | 105014 | 1.0000 | 0.0769 | 8078.9231 | 0.0 | 0.0 |
| CUSUM | `29_sampling_cusum_20260802_20260802T151114Z` | 53293 | 0.5075 | 0.1514 | 8067.3846 | 0.0 | 0.0 |
| Volatility CUSUM | `29_sampling_volatility_cusum_20260802_20260802T151251Z` | 35247 | 0.3356 | 0.2279 | 8031.3077 | 0.0 | 0.0 |
| Edge triggered | `29_sampling_edge_20260802_20260802T151420Z` | 45454 | 0.4328 | 0.1710 | 7771.2308 | 0.0 | 0.0 |

Decision: close #29 as experiment complete with no promotion. Event sampling improves uniqueness, especially volatility-CUSUM, but the frozen meta-label policy still produces zero validation/test trades.

## Parent Issue Notes

When updating parent issues, use this interpretation:

- #24 and #23 have enough regenerated evidence to support closing #25 and #26 as completed experiment gates.
- #8, #9, and #10 should remain open unless their implementation checklists are otherwise complete. The #27-#29 experiment gates are completed, but their outcomes do not promote a new bar clock, feature set, or sampling mode.
- Follow-up work should move to #14, #15, #13, #21, #11, #12, and #18 before spending more cycles on threshold-only backtests.
