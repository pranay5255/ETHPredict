# Grader v1 baseline measurements

This initial measurement is retained pending a final repeat after audit-probe
hardening. The final audit found that policy-visible seeds made the old probe
locations predictable. The corrected parent uses independent entropy after the
full reference prediction is committed; the same six adapters are being rerun.

All six predeclared adapters ran on the full public ETH dev population at the common 1h clock, with five purged growing-window evaluation blocks and both grader seeds. No parameters were selected for profitability. Shadow and holdout were not evaluated.

Execution-data SHA-256: `d452fdadad26d073b2a9618e00a56d087645e456918aaef79db8170aacae930f`. Grading fingerprint: `1b016f28c72ad25282d02439abba11e1c2ce86221c62eeac7457e7ffbe6a9f4a`.
The economic comparison winner is `flat`, chosen by mean net PnL across the two seeds. Every economic gate remains strict. Final rejected scores are zero; the separately reported pre-gate components are the approved calibration measure.

The initial anchors already meet the 0.1–0.3 calibration target and are retained,
along with every weight and gate. Only the calibration state is marked complete.
The pinned-checkpoint publication annotation was corrected after measurement;
the [correction receipt](grader_baseline_metadata_correction.json) proves that
all numeric-score/resource digests and covered artifacts remained unchanged.
See [checkpoint provenance](grader_checkpoint_provenance.md) for the October 3
UTC evidence boundary used by the final labels.

| Baseline | Pre-gate score | Final score | Net 1x, seed 17 (bps) | Net 1x, seed 29 (bps) | Worker seconds | Peak VRAM (GB) | Peak RSS (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 0.100000 | 0.000000 | 0.000 | 0.000 | 76.533 | 0.470 | 0.800 |
| buy_and_hold | 0.150015 | 0.000000 | -4057.707 | -4057.707 | 78.633 | 0.470 | 0.799 |
| momentum_1h | 0.150000 | 0.000000 | -29285.792 | -29285.792 | 112.821 | 0.470 | 0.802 |
| momentum_24h | 0.150003 | 0.000000 | -6847.241 | -6847.241 | 113.909 | 0.470 | 0.802 |
| timesfm_zero_shot | 0.150000 | 0.000000 | -26107.956 | -26107.956 | 496.417 | 1.571 | 2.674 |
| lstm_recipe | 0.173941 | 0.000000 | -11.865 | 0.000 | 1258.437 | 0.648 | 1.383 |

GB denotes decimal bytes, matching the 22 GB/48 GB gates. Worker times include fitting, prediction, model reloads and all integrity/permutation reruns across both seeds. Trusted data preparation, accounting, bootstrap and artifact hashing are outside the fit/predict budget. Per-block times below sum the actual audited worker calls and exclude the one shared declaration call.

## flat

Budget: `fast`. Status: `rejected`. Calls: 225. Report: `077096e781be99eb2fc01a05cf924e73cfec180b3d8f29fab4f743bf8dcd5e88`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 8.116 |
| 17 | 2 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.456 |
| 17 | 3 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.439 |
| 17 | 4 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.486 |
| 17 | 5 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.422 |
| 29 | 1 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 8.258 |
| 29 | 2 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.500 |
| 29 | 3 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.482 |
| 29 | 4 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.529 |
| 29 | 5 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 7.524 |

Failed checks/gates: economic:beats_best_baseline, economic:block_trade_coverage, economic:enough_trades, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash, statistics:available.

## buy_and_hold

Budget: `fast`. Status: `rejected`. Calls: 225. Report: `eed475bd992edf13a9ba4fd8beb88106b194ca6a9aad7a7bbddb420e12617b29`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | 1145.636 | 952.326 | 1 | 1.038 | 2.000 | 8.261 |
| 17 | 2 | -2545.125 | -2676.115 | 1 | -2.238 | 2.000 | 7.702 |
| 17 | 3 | -85.441 | -238.501 | 1 | -0.097 | 2.000 | 7.552 |
| 17 | 4 | -4086.891 | -4184.871 | 1 | -3.043 | 2.000 | 7.741 |
| 17 | 5 | 1514.113 | 1445.553 | 1 | 1.793 | 2.000 | 7.755 |
| 29 | 1 | 1145.636 | 952.326 | 1 | 1.038 | 2.000 | 8.410 |
| 29 | 2 | -2545.125 | -2676.115 | 1 | -2.238 | 2.000 | 7.698 |
| 29 | 3 | -85.441 | -238.501 | 1 | -0.097 | 2.000 | 7.680 |
| 29 | 4 | -4086.891 | -4184.871 | 1 | -3.043 | 2.000 | 7.779 |
| 29 | 5 | 1514.113 | 1445.553 | 1 | 1.793 | 2.000 | 7.736 |

Failed checks/gates: economic:beats_best_baseline, economic:block_trade_coverage, economic:enough_trades, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash.

## momentum_1h

Budget: `fast`. Status: `rejected`. Calls: 325. Report: `db783bc553024afcf376838184e5b5dc0eccd27947e69548ef70ab16f0fa908b`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | -6735.559 | -14388.689 | 687 | -7.043 | 1374.000 | 11.688 |
| 17 | 2 | -5133.135 | -12605.775 | 672 | -4.933 | 1344.000 | 11.021 |
| 17 | 3 | -7648.618 | -15370.568 | 695 | -8.767 | 1390.000 | 11.056 |
| 17 | 4 | -2132.193 | -9624.923 | 674 | -1.904 | 1348.000 | 11.320 |
| 17 | 5 | -7636.286 | -15301.876 | 692 | -8.888 | 1384.000 | 10.919 |
| 29 | 1 | -6735.559 | -14388.689 | 687 | -7.043 | 1374.000 | 11.765 |
| 29 | 2 | -5133.135 | -12605.775 | 672 | -4.933 | 1344.000 | 11.264 |
| 29 | 3 | -7648.618 | -15370.568 | 695 | -8.767 | 1390.000 | 11.200 |
| 29 | 4 | -2132.193 | -9624.923 | 674 | -1.904 | 1348.000 | 11.113 |
| 29 | 5 | -7636.286 | -15301.876 | 692 | -8.888 | 1384.000 | 11.144 |

Failed checks/gates: economic:beats_best_baseline, economic:credible_features, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash.

## momentum_24h

Budget: `fast`. Status: `rejected`. Calls: 325. Report: `4779a07a26b158de37579a5b81f7a70683fb1b7498055c1857b2c56db3fdb976`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | -816.047 | -2467.927 | 141 | -0.726 | 282.000 | 11.862 |
| 17 | 2 | 517.744 | -831.446 | 116 | 0.479 | 232.000 | 11.137 |
| 17 | 3 | -7764.929 | -9516.499 | 152 | -7.609 | 304.000 | 11.197 |
| 17 | 4 | 2331.718 | 1128.908 | 101 | 1.827 | 202.000 | 11.397 |
| 17 | 5 | -1115.727 | -2557.737 | 125 | -1.448 | 250.000 | 11.189 |
| 29 | 1 | -816.047 | -2467.927 | 141 | -0.726 | 282.000 | 11.846 |
| 29 | 2 | 517.744 | -831.446 | 116 | 0.479 | 232.000 | 11.237 |
| 29 | 3 | -7764.929 | -9516.499 | 152 | -7.609 | 304.000 | 11.137 |
| 29 | 4 | 2331.718 | 1128.908 | 101 | 1.827 | 202.000 | 11.280 |
| 29 | 5 | -1115.727 | -2557.737 | 125 | -1.448 | 250.000 | 11.314 |

Failed checks/gates: economic:beats_best_baseline, economic:credible_features, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash.

## timesfm_zero_shot

Budget: `long`. Status: `rejected`. Calls: 225. Report: `368f732c8c0cc16c995b559ca1180bc9020fd1925f877c1c6c077c957c2d5799`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | -3317.124 | -7226.974 | 347 | -4.334 | 694.000 | 53.834 |
| 17 | 2 | -7060.161 | -11148.911 | 365 | -6.612 | 730.000 | 46.705 |
| 17 | 3 | -3751.577 | -7848.667 | 366 | -4.541 | 732.000 | 48.604 |
| 17 | 4 | -6573.417 | -10587.457 | 360 | -5.242 | 720.000 | 49.900 |
| 17 | 5 | -5405.677 | -9330.417 | 353 | -6.751 | 706.000 | 50.039 |
| 29 | 1 | -3317.124 | -7226.974 | 347 | -4.334 | 694.000 | 51.690 |
| 29 | 2 | -7060.161 | -11148.911 | 365 | -6.612 | 730.000 | 47.564 |
| 29 | 3 | -3751.577 | -7848.667 | 366 | -4.541 | 732.000 | 49.520 |
| 29 | 4 | -6573.417 | -10587.457 | 360 | -5.242 | 720.000 | 48.340 |
| 29 | 5 | -5405.677 | -9330.417 | 353 | -6.751 | 706.000 | 49.714 |

Failed checks/gates: economic:beats_best_baseline, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash.

Checkpoint licenses and per-block evaluation-period labels:

- Block 1: `timesfm-2.5`, `Apache-2.0`; possibly in pretraining.
- Block 2: `timesfm-2.5`, `Apache-2.0`; possibly in pretraining.
- Block 3: `timesfm-2.5`, `Apache-2.0`; after documented public release.
- Block 4: `timesfm-2.5`, `Apache-2.0`; after documented public release.
- Block 5: `timesfm-2.5`, `Apache-2.0`; after documented public release.

Pre-release fitting periods are labeled separately in the full reports. Release timing does not establish a complete pretraining corpus or an independent holdout.

## lstm_recipe

Budget: `long`. Status: `rejected`. Calls: 445. Report: `57fbb0ba84caf81e344534595d9d1e1d4a987b11e85218bf070d3a7ae2402d46`.

| Seed | Block | Net 1x (bps) | Net 2x (bps) | Trades | Daily net Sharpe (annualized) | Turnover | Worker seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | -11.865 | -45.225 | 3 | -0.493 | 6.000 | 140.587 |
| 17 | 2 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 103.901 |
| 17 | 3 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 124.313 |
| 17 | 4 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 136.631 |
| 17 | 5 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 139.298 |
| 29 | 1 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 119.247 |
| 29 | 2 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 110.631 |
| 29 | 3 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 131.315 |
| 29 | 4 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 118.434 |
| 29 | 5 | 0.000 | 0.000 | 0 | unavailable | 0.000 | 133.589 |

Failed checks/gates: economic:beats_best_baseline, economic:block_trade_coverage, economic:credible_features, economic:enough_trades, economic:positive_across_blocks, economic:positive_net_expectancy, economic:survives_cost_stress, grader_hash, statistics:available.

## Evidence and interpretation

The private-file boundary and code-only agent sandbox are verified separately. Baseline reports, positions, split audits, feature diagnostics, opaque fitted checkpoints, syscall traces and artifact manifests remain in the durable local dev bank. Reports retain both execution provenance and all opted-in input provenance. Undefined Sharpe values are reported as unavailable and fail closed; they are never replaced by invented observations.

Before the tag exists, the `grader_hash` check records the expected pre-freeze failure. All other integrity checks must pass for this bank to be complete. An economic-gate rejection is not a hidden test result or an authorization to trade.
