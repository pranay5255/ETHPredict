# Changelog

## 2026-10-17 — local data, documentation and EDA update

- Record completed five-market mainnet monthly Parquet collection through a
  partial October 16: 25 readable datasets, 425 matching partition hashes and
  303.10 MB (289.06 MiB). Keep the lake ignored in Git.
- Preserve the existing backfill collector, shared-schema repair and notebook/EDA
  work, with one file per commit for traceability.
- Correct the funding interpretation: 31 shared missing hours are unresolved
  collection gaps; six exist in the older ETH archive. Recovery tracked in #54.
- Refresh active status, task/readiness/strategy plans, data/dependency/dashboard
  guides and visual roadmap. Mark older reviews/results with their historical scope.
- Distinguish unmerged PR #51 (#16/#17/#20) and stacked draft PR #53 from main.
  Record their negative bounded studies without changing scientific artifacts or
  consuming another final test. No existing open issue qualifies for closure from
  backfill alone; GitHub now has 21 open issues including new #54.
- Training remains CSV-based; general lake ingestion, sizing #19, benchmark parity
  #22 and current-state #43–#45 remain work. Verification for this local series is
  recorded in `context/project_docs_verification_20261017.md`.


## v0.6 - AFML validity, accounting, and evidence

- Added label-span uniqueness sample weights and effective-sample diagnostics (#14, PRs #42/#46).
- Made Trackio accounting mandatory for non-smoke research runs, with local readback, delivery receipts, and persisted trial accounting (#23, PR #47).
- Scored the final test only after validation selection, counted forecast-benchmark test evaluations, and graded evidence from the final-test ledger (#24, PR #47).
- Added Sharpe, PSR, DSR, PBO, underwater span, HHI, and evidence grades to backtest statistics; CSCV winners are ranked against the out-of-sample median (#21, PR #47).
- Prevented future-data leakage in v2 transforms and made preprocessing fold-local (#9, PR #48).
- Made triple-barrier success definitions, geometry, and dual-touch ambiguity policy configurable and auditable (#13, PR #49).
- Purged and embargoed by exact label spans, added purged k-fold and grouped scenarios, and nested base predictions for held-out meta evaluation (#15, PR #49).
- Added config-selected historical side-data joins with publication-time alignment, source hashes, and per-split coverage gates; no group is enabled by default (#8, PR #49).
- Archived Trackio runs predate PR #48 and #49 and are not evidence for this version's code.

## v0.5 - uv CUDA 12.8 Lighter experiments

- Imported the current dependency set into `pyproject.toml` and `uv.lock` with uv.
- Kept `requirements.txt` unchanged for legacy setup while making `pyproject.toml` and `uv.lock` the active experiment dependency source.
- Configured PyTorch to use the official CUDA 12.8 `pytorch-cu128` wheel index on Linux/Windows.
- Replaced CUDA 13 runtime artifacts in the lock/environment with CUDA 12.x packages for the RTX 4090 experiment path.
- Verified `torch==2.11.0+cu128` with CUDA available on `NVIDIA GeForce RTX 4090`.
- Added GPU-first device resolution for neural training, with explicit CPU override via `allow_cpu`/`--allow-cpu`.
- Added `src.experiments.lighter_compare` for Lighter-only neural trials, ARIMA/SARIMAX CPU baselines, and GLFT metric ranking.
- Added `configs/lighter_experiments.yml` for triple-barrier and next-hour-return experiment targets.
- Added smoke tests for uv Torch CUDA visibility, one-batch GPU training for `PriceLSTM`, `MetaMLP`, and `ConfidenceGRU`, CPU ARIMA/SARIMAX baselines, and end-to-end Lighter compare smoke execution.
- Verified `uv run pytest`, `uv lock --check`, and `uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml --smoke`.

## v0.4 - Lighter-only data scope

- Switched active data configuration to public Lighter ETH perp market data only.
- Collected and validated `data/raw/ETHUSDT-1h-lighter-20260328-20260628.csv` for the current raw pass.
- Refactored `DataPreprocessor` to load only `*-lighter-*` raw CSV files and generate OHLCV-only features.
- Archived legacy Binance, DeFiLlama, Santiment, DEX simulation, bribe optimization, and parameter optimization assets under `archive/legacy_data_sources`.
- Removed active `bribe`, `sim`, and `optimization` sections from `configs/config.yml` and active schema validation.
- Simplified config loading exports for the active Lighter-only runner path.
- Renamed active Lighter CSV writer helpers from Binance-compatible wording to model OHLCV wording.
- Added focused tests for Lighter client normalization and Lighter-only preprocessing.
- Verified focused tests, py_compile checks, and Lighter feature-build smoke output to `/tmp/ethpredict-lighter-features`.

## v0.3 - Unified pipeline

- End-to-end pipeline entry point `src.runner`.
- CSV validation, bar sampling, feature generation.
- Simple PyTorch/Ray training.
- Backtesting with GLFT market maker.
- Docker compose and CI workflow.
