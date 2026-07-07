# AFML Freeze Experiment Report

Generated from local GPU freeze runs for GitHub issues #25-#29. All experiment artifacts are under `/tmp/ethpredict-afml-freeze`; derived reports are under `/tmp/ethpredict-afml-freeze/reports`.

## Executive Result

- Local CUDA execution works on the RTX 4090 path; manifests record `cuda:0`, `torch 2.11.0+cu128`, and raw data hashes.
- Reproducibility controls pass for identical frozen reruns and fail as expected for a controlled config change while raw data and split hashes stay fixed.
- Trial accounting now covers completed, failed, skipped, no-trade, low-trade, and trade-qualified buckets.
- The current frozen 0.55 meta-label policy produces zero validation/test trades across #27-#29. This validates instrumentation, not tradable alpha.
- The #26 forced lower-threshold trade case generated trades but had negative validation/test PnL, so threshold loosening alone is not a model fix.

## Issue #25: Reproducibility Gate
| Run | Run id | Final-test reuse count | Validation trades |
| --- | --- | --- | --- |
| baseline A | 25_repro_baseline_20260707T200437Z | 1 | 0.0 |
| baseline B | 25_repro_baseline_20260707T200648Z | 2 | 0.0 |
| changed config | 25_config_changed_20260707T200900Z | 1 | 0.0 |

Baseline rerun comparison: `pass`. Changed-config control: `fail` with `config_hash_match=false` and `raw_hashes_match=true`.

## Issue #26: Trial Accounting

Run: `26_accounting_matrix_v3_20260707T202829Z`

Counts: `{'completed': 3, 'failed': 1, 'low_trade': 2, 'no_trade': 1, 'skipped': 1, 'trade_qualified': 1}`

| Trial | Status | Selection roles | Validation trades | Test trades |
| --- | --- | --- | --- | --- |
| normal_trade | completed | trade_qualified_best, calibration_best | 14121.0 | 8164.0 |
| low_trade | completed | not_selected | 8.0 | 0.0 |
| no_trade | completed | raw_best, classification_best | 0.0 | 0.0 |
| skipped_control | skipped | not_selected | 0.0 | 0.0 |
| failed_missing_edge | failed | not_selected | 0.0 | 0.0 |

## Issue #27: Bar Clock
| Clock | Rows | Coverage | Median interval sec | Log-return std | Lag-1 corr | Val trades |
| --- | --- | --- | --- | --- | --- | --- |
| time | 105074 | 1.0 | 300.0 | 0.002 | -0.001799 | 0.0 |
| volume | 10036 | 0.096 | 2400.0 | 0.006 | -0.003356 | 0.0 |
| dollar | 10222 | 0.097 | 2100.0 | 0.006 | 0.001646 | 0.0 |

Activity bars reduced sample count to roughly 10% of 5-minute bars and changed volatility distribution, but side-data groups stayed disabled and all frozen policies remained no-trade at 0.55.

## Issue #28: Feature Ablation
| Variant | Inputs | Families | Fracdiff mode | ADF p | Memory corr | Val trades | Scaler pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ohlcv_only | 6 | 1 | none | None | None | 0.0 | True |
| full_no_fracdiff | 23 | 8 | none | None | None | 0.0 | True |
| full_fixed_fracdiff | 24 | 9 | fixed_width | 0.843631 | 0.999445 | 0.0 | True |

All variants used fold-local scaler fitting. Fixed-width fracdiff preserved near-perfect memory correlation but did not produce trades under the frozen 0.55 policy.

## Issue #29: Event Sampling
| Mode | Events | Event rate | Avg uniqueness | Effective N | Val trades | Test trades |
| --- | --- | --- | --- | --- | --- | --- |
| dense | 105014 | 1.0 | 0.077 | 8078.923 | 0.0 | 0.0 |
| cusum | 53293 | 0.507 | 0.151 | 8067.385 | 0.0 | 0.0 |
| volatility_cusum | 35247 | 0.336 | 0.228 | 8031.308 | 0.0 | 0.0 |
| edge_triggered | 45454 | 0.433 | 0.171 | 7771.231 | 0.0 | 0.0 |

CUSUM, volatility-CUSUM, and edge-triggered sampling improved uniqueness relative to dense sampling, but none solved the no-trade meta-threshold behavior.

## Generated Artifacts

- `/tmp/ethpredict-afml-freeze/reports/manifest_diff.json`
- `/tmp/ethpredict-afml-freeze/reports/reproducibility_report.md`
- `/tmp/ethpredict-afml-freeze/reports/trial_accounting.json`
- `/tmp/ethpredict-afml-freeze/reports/failure_mode_panels.json`
- `/tmp/ethpredict-afml-freeze/reports/trackio_parity_report.md`
- `/tmp/ethpredict-afml-freeze/reports/bar_clock_comparison.json`
- `/tmp/ethpredict-afml-freeze/reports/side_data_coverage_report.md`
- `/tmp/ethpredict-afml-freeze/reports/feature_ablation_matrix.json`
- `/tmp/ethpredict-afml-freeze/reports/fracdiff_diagnostics.json`
- `/tmp/ethpredict-afml-freeze/reports/scaler_leakage_report.md`
- `/tmp/ethpredict-afml-freeze/reports/event_sampling_comparison.json`
- `/tmp/ethpredict-afml-freeze/reports/event_rate_report.md`
- `/tmp/ethpredict-afml-freeze/reports/afml_freeze_summary.json`

## Modeling Follow-Up

The next useful work is not more threshold-only backtesting. The evidence points at the open modeling follow-ups: forecast calibration/candidate mapping (#11), training discipline and baselines (#12), meta-labeler calibration/threshold policy (#18), and probability-sized bet sizing (#22).
