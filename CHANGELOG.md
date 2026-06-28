# Changelog

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
