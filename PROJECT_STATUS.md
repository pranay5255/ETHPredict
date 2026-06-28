# ETHPredict Project Status

## Current Scope

ETHPredict is in a Lighter-only raw market-data phase with a uv-managed GPU experiment path. Active code uses public Lighter ETH perpetual OHLCV data for feature generation, runs neural experiments on `cuda:0`, and keeps non-Lighter sources archived for later recovery.

Active source:

- Lighter mainnet ETH perp, `market_id: 0`
- Raw model CSV: `data/raw/ETHUSDT-1h-lighter-20260328-20260628.csv`
- Side-data archive: `data/lighter/`

Active experiment path:

- Dependency source: `pyproject.toml` and `uv.lock`
- Neural device default: `cuda:0`
- Verified GPU: `NVIDIA GeForce RTX 4090`
- Torch runtime: `2.11.0+cu128`
- Experiment runner: `uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml`

Archived scope:

- Binance historical OHLCV samples
- DeFiLlama TVL snapshots
- Santiment social/network metrics
- DEX simulation config/code notes
- Bribe/MEV optimization config/code notes
- Parameter optimization config/code notes

Archive location: `archive/legacy_data_sources/`

## Implemented and Active

### UV GPU Experiment Environment

Status: active for neural experiments on the 4090 machine.

- `pyproject.toml` and `uv.lock` now contain the active dependency set imported from `requirements.txt`.
- `requirements.txt` remains unchanged for legacy setup until a future cleanup.
- PyTorch is configured for Linux/Windows through the official `pytorch-cu128` index with `torch>=2.11.0`.
- Verified Torch runtime: `2.11.0+cu128`, CUDA available, device `NVIDIA GeForce RTX 4090`.
- `src/training/devices.py` centralizes device resolution.
- Neural training defaults to `cuda:0` and fails fast when CUDA is unavailable unless CPU execution is explicitly allowed.

### Lighter Data Collection

Status: complete for the current raw pass.

- `scripts/lighter_collect_data.py` collects public candles, mark-price candles, fundings, funding-rate snapshots, order-book details/orders, and recent trades.
- No Lighter auth or trading keys are required for the active collection pass.
- Optional `exchangeMetrics` outputs remain best-effort.

### Lighter OHLCV Feature Generation

Status: active.

- `src/data/features_all.py` now reads only `data/raw/ETHUSDT-<resolution>-lighter-*.csv`.
- Default granularity is `1h`.
- `include_santiment` remains accepted only as a compatibility no-op.
- Feature generation uses OHLCV-derived fields only: price/volume, quote volume, returns, range, volatility, entropy, CUSUM/SADF flags, volatility regime, Parkinson volatility, and fractional-diff close.
- Targets are active market fields: `close` and `volume`; sequence targets use next-step normalized close.

### Lighter Compare Experiments

Status: smoke-tested and active as the uv experiment entry point.

- `configs/lighter_experiments.yml` defines two active targets: `triple_barrier` and `next_hour_return`.
- `src.experiments.lighter_compare` runs the `PriceLSTM`, `MetaMLP`, and `ConfidenceGRU` stack on `cuda:0`.
- The runner includes one forward/backward stack smoke, one or more neural trials per target, CPU ARIMA/SARIMAX baselines, and GLFT backtest metrics for finalist ranking.
- Smoke mode currently uses 104 sequence samples per target from the active Lighter OHLCV file.
- ARIMA and SARIMAX baselines remain CPU-bound.

### Configuration

Status: active schema and loader simplified.

- `configs/config.yml` is Lighter-only for the top-level pipeline.
- `configs/lighter_experiments.yml` is the active uv experiment config.
- `configs/schema.yaml` no longer models active `bribe` or `sim` sections.
- `src/config/loader.py` validates the active config and exposes a compact typed config object.
- `src/config/__init__.py` now exports the active config loader symbols instead of non-existent sweep/core modules.

### Archive

Status: complete for this refactor.

- Legacy CSVs were moved into `archive/legacy_data_sources/data`.
- Legacy multi-source code/config/docs were copied into `archive/legacy_data_sources/code` and `archive/legacy_data_sources/docs`.
- Active loaders do not read from the archive.

### Tests and Smoke Checks

Status: full uv test suite and focused smoke checks passing.

Verified commands:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
uv run pytest
uv run pytest tests/test_lighter_experiments.py
uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml --smoke
uv lock --check
python -m pytest tests/test_lighter_client.py tests/test_lighter_preprocessor.py tests/test_config.py tests/test_runner.py
python -m py_compile src/data/features_all.py src/data/lighter_client.py scripts/lighter_collect_data.py src/config/loader.py src/config/__init__.py src/main.py runner.py
python -m src.data.features_all full_features --data-dir data --sequence-length 24 --out-dir /tmp/ethpredict-lighter-features
```

## Partially Implemented / Needs Work

### End-to-End Pipeline Run

Status: needs verification after the refactor.

- `runner.py` now initializes the Lighter-only preprocessor.
- Backtesting still has fallback behavior using shifted prices if model prediction integration fails.
- Next step: run `python runner.py configs/config.yml` and fix any integration issues from model/training/backtest boundaries.

### Model Prediction Integration in Backtesting

Status: partial.

- The code attempts model prediction when possible.
- The fallback shifted-price path still exists.
- Next step: make model inference the primary path and fail clearly if model outputs cannot be produced.

### Full Lighter Compare Experiment

Status: smoke-tested; full run still pending.

- Smoke mode has verified the end-to-end experiment plumbing on `cuda:0`.
- The non-smoke configured experiment should be run next.
- Finalist review should tune prediction metrics first, then rank by GLFT backtest metrics.

### Lighter Side-Data Features

Status: collected but not integrated.

- Funding, mark price, order book, and recent trades are available under `data/lighter/`.
- Active features intentionally ignore side data for now.
- Recommended next addition: funding and mark price joins, because they have historical time indexes.

### Training Environment

Status: GPU-first policy implemented for neural experiments.

- The Lighter feature preprocessor remains Lighter/OHLCV scoped.
- Model/training modules depend on the uv-managed Torch environment.
- Neural training fails fast without CUDA unless CPU use is explicitly allowed.
- A future policy decision is still needed on whether CPU neural fallback should be documented beyond tests/development.

### Documentation

Status: active docs updated for Lighter-only scope and uv GPU experiments.

- README, TASKS, PROJECT_STATUS, CHANGELOG, and Lighter context docs describe the active Lighter-only path.
- README, TASKS, PROJECT_STATUS, and CHANGELOG now also describe the uv CUDA 12.8 experiment path.
- Older broad research docs remain in `context/` as planning references unless explicitly archived later.

## Not Active in This Phase

- Binance, DeFiLlama, and Santiment feature joins.
- DEX simulation.
- Bribe/MEV optimization.
- Parameter sweeps and Bayesian optimization.
- Signed Lighter trading or live execution.
- Production deployment service.

## Completion Snapshot

| Area | Status |
| --- | --- |
| Lighter data collection | Complete for raw pass |
| Raw OHLCV validation | Complete for current file |
| Lighter-only feature generation | Active and smoke-tested |
| uv CUDA 12.8 dependency path | Active and verified |
| CUDA 4090 Torch validation | Passing |
| Lighter compare smoke runner | Passing |
| Legacy source archive | Complete |
| Config simplification | Complete |
| Full uv test suite | Passing |
| End-to-end runner | Needs post-refactor verification |
| Model-backed backtest predictions | Partial |
| Full Lighter compare experiment | Pending |
| Lighter side-data joins | Not started |
| Live trading/execution | Out of scope |

## Next Recommended Steps

1. Run the top-level pipeline with the active config:

   ```bash
   python runner.py configs/config.yml
   ```

2. Fix any model/training/backtest integration failures exposed by that run.
3. Run the non-smoke Lighter compare experiment through uv and inspect the ranked finalists.
4. Replace shifted-price fallback predictions with explicit model inference outputs.
5. Decide whether to add funding and mark-price side-data joins.
6. Keep archived legacy sources untouched unless a new experiment explicitly reopens them.
