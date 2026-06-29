# ETHPredict Lighter-Only Task List

## Completed in Lighter-Only Refactor

- [x] Set `configs/config.yml` to `data.sources: [lighter]`.
- [x] Configure Lighter ETH perp mainnet market `market_id: 0` with `1h` resolution.
- [x] Collect raw Lighter OHLCV for `2026-03-28` through `2026-06-28`.
- [x] Validate raw Lighter OHLCV row count, timestamp ordering, uniqueness, and OHLC numeric quality.
- [x] Archive legacy Binance, DeFiLlama, and Santiment CSV inputs under `archive/legacy_data_sources/data`.
- [x] Archive legacy multi-source feature/config/docs/code snapshots under `archive/legacy_data_sources`.
- [x] Remove active bribe optimization, DEX simulation, and parameter optimization config sections.
- [x] Refactor `DataPreprocessor` to load only `data/raw/*-lighter-*.csv` and build OHLCV-only features.
- [x] Add tests proving legacy-looking raw files are ignored.
- [x] Verify focused tests and Lighter feature-build smoke command.

## Completed in UV GPU Experiment Setup

- [x] Import the current `requirements.txt` dependency set into the uv project metadata.
- [x] Keep `requirements.txt` unchanged for legacy setup while treating `pyproject.toml` and `uv.lock` as the active experiment dependency source.
- [x] Configure `torch>=2.11.0` to use the official PyTorch CUDA 12.8 wheel index on Linux/Windows.
- [x] Refresh `uv.lock` so CUDA 13 artifacts are replaced with CUDA 12.x packages for the 4090 path.
- [x] Run `uv sync` for the active environment.
- [x] Verify uv-managed Torch reports `2.11.0+cu128`, CUDA available, and `NVIDIA GeForce RTX 4090` on `cuda:0`.
- [x] Add shared CUDA device resolution that defaults neural training to `cuda:0` and requires `--allow-cpu` for CPU neural runs.
- [x] Add `src.experiments.lighter_compare` for Lighter-only neural trials, CPU ARIMA/SARIMAX baselines, and GLFT metric ranking.
- [x] Add `configs/lighter_experiments.yml` for triple-barrier and next-hour-return experiment targets.
- [x] Add small full-data staged configs for 5m `next_hour_return` and `triple_barrier` experiments.
- [x] Run exploratory small staged 5m experiments and confirm serial full-data execution succeeds after parallel CUDA OOM fallback.
- [x] Add CUDA environment, one-batch neural stack, CPU baseline, and end-to-end smoke tests.
- [x] Verify `uv run pytest`, `uv lock --check`, and the Lighter compare smoke command.

## Active Near-Term Tasks

- [ ] Add `next_5m_return` and multi-horizon model outputs for 5m and 1h returns/direction.
- [ ] Add purged walk-forward CV that stores out-of-sample base predictions for each fold.
- [ ] Implement triple-barrier meta-label generation for proposed long/short signals using profit-taking, stop-loss, and vertical barriers after estimated costs.
- [ ] Train a true meta-label classifier from out-of-sample base predictions, replacing or separating the current `y_dir != 0` confidence target.
- [ ] Report meta-label threshold performance: coverage, directional accuracy, hit ratio, gross/net PnL, fees, turnover, and drawdown.
- [ ] Replace Stage 2's GLFT-first ranking with a validation-selected directional alpha backtest.
- [ ] Add Stage 0 funding and mark-price joins first; add historical order-book/trade-flow features only when time-indexed history is available.
- [ ] Keep GLFT as an optional passive execution layer after alpha validation, using policy target inventory or reservation-price skew.
- [ ] Run `python runner.py configs/config.yml` end to end after deciding whether current model/training changes should be kept.
- [ ] Add a small fixture-backed integration test for `runner.py` using Lighter-only raw data.
- [ ] Decide whether to migrate legacy `requirements.txt` users fully to uv in a later cleanup.

## Deferred / Archived for Later

- [ ] Reintroduce Binance OHLCV only if a mixed-source experiment is explicitly reopened.
- [ ] Reintroduce DeFiLlama/Santiment joins only after defining target/feature semantics for non-market data.
- [ ] Reintroduce DEX simulation only after the price-data backtest path is stable.
- [ ] Treat GLFT parameter search as deferred alpha evaluation; use it only for passive execution research after directional alpha validation.
- [ ] Reintroduce bribe/MEV optimization only as a separate execution research track.
- [ ] Reintroduce parameter sweeps only after the base Lighter-only run is reproducible.
- [ ] Add signed Lighter trading or live execution only after offline experiments are reproducible.

## Useful Commands

```bash
python scripts/lighter_collect_data.py --config configs/config.yml
python -m src.data.features_all full_features --data-dir data --sequence-length 24 --out-dir /tmp/ethpredict-lighter-features
uv sync
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
uv run pytest
uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml --smoke
uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml
uv run python -m src.experiments.staged_trial --config configs/staged_trial_small_next_hour.yml --run-name staged_5m_small_next_hour
uv run python -m src.experiments.staged_trial --config configs/staged_trial_small_triple_barrier.yml --run-name staged_5m_small_triple_barrier
python runner.py configs/config.yml
```
