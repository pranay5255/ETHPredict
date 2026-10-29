# Grader v1 baseline measurements

This is the final repeat after causality audit locations were separated from policy-visible seeds. All six predeclared adapters ran on the full public ETH dev population at the common 1h clock, with five purged growing-window evaluation blocks and both grader seeds. No parameters were selected for profitability. Shadow and holdout were not evaluated.

Execution-data SHA-256: `d452fdadad26d073b2a9618e00a56d087645e456918aaef79db8170aacae930f`. Grading fingerprint: `5e6f108809b25fa83379389546d7627a8e791ad8e4248964745aae606cee083e`.
The economic comparison winner is `flat`, chosen by mean net PnL across the two seeds. Every economic gate remains strict. Initial anchors and weights are retained because they already satisfy the approved calibration range. Final rejected scores are zero; the separately reported pre-gate components are the approved calibration measure.

| Baseline | Pre-gate score | Final score | Net 1x, seed 17 (bps) | Net 1x, seed 29 (bps) | Worker seconds | Peak VRAM (GB) | Peak RSS (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 0.100000 | 0.000000 | 0.000 | 0.000 | 77.891 | 0.470 | 0.800 |
| buy_and_hold | 0.150015 | 0.000000 | -4057.707 | -4057.707 | 78.276 | 0.470 | 0.799 |
| momentum_1h | 0.150000 | 0.000000 | -29285.792 | -29285.792 | 114.509 | 0.470 | 0.803 |
| momentum_24h | 0.150003 | 0.000000 | -6847.241 | -6847.241 | 114.167 | 0.470 | 0.803 |
| timesfm_zero_shot | 0.150000 | 0.000000 | -26107.956 | -26107.956 | 494.895 | 1.571 | 2.674 |
| lstm_recipe | 0.173941 | 0.000000 | -11.865 | 0.000 | 1165.877 | 0.648 | 1.384 |

GB denotes decimal bytes, matching the 22 GB/48 GB gates. Worker times include fitting, prediction, model reloads and all integrity/permutation reruns across both seeds. Trusted data preparation, accounting, bootstrap and artifact hashing are outside the fit/predict budget. Per-block times below sum the actual audited worker calls and exclude the one shared declaration call.

## flat

Budget: `fast`. Status: `rejected`. Calls: 225. Report: `d06e65f024da766813c25f6899e22e0fe70bb3faf96bf510409d76b918c7f49a`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 8.185 |
| 17 | 2 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.607 |
| 17 | 3 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.639 |
| 17 | 4 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.596 |
| 17 | 5 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.649 |
| 29 | 1 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 8.267 |
| 29 | 2 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.641 |
| 29 | 3 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.658 |
| 29 | 4 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.670 |
| 29 | 5 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.663 |

Failed checks/gates: economic:beats_best_baseline, economic:block_trade_coverage, economic:enough_trades, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash, statistics:available.

## buy_and_hold

Budget: `fast`. Status: `rejected`. Calls: 225. Report: `7b30c671812b0012c9fb536844cc66866ea3c0e4d2af6bdc2ffd30a498b2f5b2`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | 1145.636 | 952.326 | 1 | 1.038 | 2.000 | 8.379 |
| 17 | 2 | -2545.125 | -2676.115 | 1 | -2.238 | 2.000 | 7.489 |
| 17 | 3 | -85.441 | -238.501 | 1 | -0.097 | 2.000 | 7.709 |
| 17 | 4 | -4086.891 | -4184.871 | 1 | -3.043 | 2.000 | 7.735 |
| 17 | 5 | 1514.113 | 1445.553 | 1 | 1.793 | 2.000 | 7.610 |
| 29 | 1 | 1145.636 | 952.326 | 1 | 1.038 | 2.000 | 8.376 |
| 29 | 2 | -2545.125 | -2676.115 | 1 | -2.238 | 2.000 | 7.825 |
| 29 | 3 | -85.441 | -238.501 | 1 | -0.097 | 2.000 | 7.578 |
| 29 | 4 | -4086.891 | -4184.871 | 1 | -3.043 | 2.000 | 7.637 |
| 29 | 5 | 1514.113 | 1445.553 | 1 | 1.793 | 2.000 | 7.623 |

Failed checks/gates: economic:beats_best_baseline, economic:block_trade_coverage, economic:enough_trades, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash.

## momentum_1h

Budget: `fast`. Status: `rejected`. Calls: 325. Report: `1b01c787fc3fd017a2f091f815046cb3d2467bb6cd1f892bd8363c293abe5d61`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | -6735.559 | -14388.689 | 687 | -7.043 | 1374.000 | 11.852 |
| 17 | 2 | -5133.135 | -12605.775 | 672 | -4.933 | 1344.000 | 11.313 |
| 17 | 3 | -7648.618 | -15370.568 | 695 | -8.767 | 1390.000 | 11.271 |
| 17 | 4 | -2132.193 | -9624.923 | 674 | -1.904 | 1348.000 | 11.313 |
| 17 | 5 | -7636.286 | -15301.876 | 692 | -8.888 | 1384.000 | 11.433 |
| 29 | 1 | -6735.559 | -14388.689 | 687 | -7.043 | 1374.000 | 12.125 |
| 29 | 2 | -5133.135 | -12605.775 | 672 | -4.933 | 1344.000 | 11.238 |
| 29 | 3 | -7648.618 | -15370.568 | 695 | -8.767 | 1390.000 | 11.248 |
| 29 | 4 | -2132.193 | -9624.923 | 674 | -1.904 | 1348.000 | 11.256 |
| 29 | 5 | -7636.286 | -15301.876 | 692 | -8.888 | 1384.000 | 11.114 |

Failed checks/gates: economic:beats_best_baseline, economic:credible_features, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash.

## momentum_24h

Budget: `fast`. Status: `rejected`. Calls: 325. Report: `c16af13ee04332758530386f76d6df94a3a4dbe1835ee6ca67574fc5f6003122`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | -816.047 | -2467.927 | 141 | -0.726 | 282.000 | 11.799 |
| 17 | 2 | 517.744 | -831.446 | 116 | 0.479 | 232.000 | 11.287 |
| 17 | 3 | -7764.929 | -9516.499 | 152 | -7.609 | 304.000 | 11.377 |
| 17 | 4 | 2331.718 | 1128.908 | 101 | 1.827 | 202.000 | 11.235 |
| 17 | 5 | -1115.727 | -2557.737 | 125 | -1.448 | 250.000 | 11.206 |
| 29 | 1 | -816.047 | -2467.927 | 141 | -0.726 | 282.000 | 12.002 |
| 29 | 2 | 517.744 | -831.446 | 116 | 0.479 | 232.000 | 11.269 |
| 29 | 3 | -7764.929 | -9516.499 | 152 | -7.609 | 304.000 | 11.185 |
| 29 | 4 | 2331.718 | 1128.908 | 101 | 1.827 | 202.000 | 11.262 |
| 29 | 5 | -1115.727 | -2557.737 | 125 | -1.448 | 250.000 | 11.213 |

Failed checks/gates: economic:beats_best_baseline, economic:credible_features, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash.

## timesfm_zero_shot

Budget: `long`. Status: `rejected`. Calls: 225. Report: `1d1a018d965788ec01b990f2e6aeb9bb05e089983a682b1251db0cea5f3f3e18`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | -3317.124 | -7226.974 | 347 | -4.334 | 694.000 | 52.116 |
| 17 | 2 | -7060.161 | -11148.911 | 365 | -6.612 | 730.000 | 49.097 |
| 17 | 3 | -3751.577 | -7848.667 | 366 | -4.541 | 732.000 | 47.902 |
| 17 | 4 | -6573.417 | -10587.457 | 360 | -5.242 | 720.000 | 48.910 |
| 17 | 5 | -5405.677 | -9330.417 | 353 | -6.751 | 706.000 | 48.200 |
| 29 | 1 | -3317.124 | -7226.974 | 347 | -4.334 | 694.000 | 53.602 |
| 29 | 2 | -7060.161 | -11148.911 | 365 | -6.612 | 730.000 | 48.824 |
| 29 | 3 | -3751.577 | -7848.667 | 366 | -4.541 | 732.000 | 48.086 |
| 29 | 4 | -6573.417 | -10587.457 | 360 | -5.242 | 720.000 | 50.414 |
| 29 | 5 | -5405.677 | -9330.417 | 353 | -6.751 | 706.000 | 47.256 |

Failed checks/gates: economic:beats_best_baseline, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash.

Checkpoint licenses and per-block evaluation-period labels:

- Block 1: `timesfm-2.5`, `Apache-2.0`; possibly in pretraining.
- Block 2: `timesfm-2.5`, `Apache-2.0`; possibly in pretraining.
- Block 3: `timesfm-2.5`, `Apache-2.0`; after documented public release.
- Block 4: `timesfm-2.5`, `Apache-2.0`; after documented public release.
- Block 5: `timesfm-2.5`, `Apache-2.0`; after documented public release.

Pre-release fitting periods are labeled separately in the full reports. Release timing does not establish a complete pretraining corpus or an independent holdout.

## lstm_recipe

Budget: `long`. Status: `rejected`. Calls: 445. Report: `0e471cb9176e38995358b0f883b297c6197b1aeb709c6710588ac52f4c74c67a`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | -11.865 | -45.225 | 3 | -0.493 | 6.000 | 126.988 |
| 17 | 2 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 103.034 |
| 17 | 3 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 118.171 |
| 17 | 4 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 126.637 |
| 17 | 5 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 102.202 |
| 29 | 1 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 116.806 |
| 29 | 2 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 115.925 |
| 29 | 3 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 102.109 |
| 29 | 4 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 125.976 |
| 29 | 5 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 127.556 |

Failed checks/gates: economic:beats_best_baseline, economic:block_trade_coverage, economic:credible_features, economic:enough_trades, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash, statistics:available.

## Evidence and interpretation

The [final validation receipt](grader_baseline_final_validation.json) compares the repeated measurements with the initial fixed-adapter run, including all 60 position files. The [checkpoint provenance review](grader_checkpoint_provenance.md) records the conservative October 3, 2025 UTC evidence boundary; publication labels were already corrected before this final repeat.

The private-file boundary and code-only agent sandbox are verified separately. Baseline reports, positions, split audits, feature diagnostics, opaque fitted checkpoints, syscall traces and artifact manifests remain in the durable local dev bank. Reports retain both execution provenance and all opted-in input provenance. Undefined Sharpe values are reported as unavailable and fail closed; they are never replaced by invented observations.

Before the tag exists, the `grader_hash` check records the expected pre-freeze failure. All other integrity checks must pass for this bank to be complete. An economic-gate rejection is not a hidden test result or an authorization to trade.
