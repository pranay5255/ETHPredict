# ETHPredict Project Status

## Current Scope

As of 2026-09-23, ETHPredict is a Lighter-only research system evolving into a local decision-support application for a discretionary trader. The planned view will show public market and feature data, model outputs with interpretation context, checkpoint comparisons, and simulated backtest evidence. The trader places real orders separately in Lighter. GitHub issue state is authoritative for the roadmap; this document distinguishes code that exists from research that is validated.

Active source:

- Lighter mainnet ETH perp, `market_id: 0`
- Raw model CSV: `data/raw/ETHUSDT-5m-lighter-20260328-20260628.csv`
- Side-data archive: `data/lighter/`

Active experiment path:

- Dependency source: `pyproject.toml` and `uv.lock`
- Neural device default: `cuda:0`
- Verified GPU: `NVIDIA GeForce RTX 4090`
- Torch runtime: `2.11.0+cu128`
- Primary v2 research runner: `uv run python -m src.experiments.staged_trial --config configs/config.yml`
- Exploratory older runner: `uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml`
- Local experiment tracking: Trackio project `ethpredict`; run artifacts under `artifacts/runs/` or a configured artifact root.

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
- Default research granularity is currently `5m` for staged trials.
- `include_santiment` remains accepted only as a compatibility no-op.
- Feature generation uses OHLCV-derived fields: price/volume, quote volume, returns, range, volatility, entropy, CUSUM/SADF flags, volatility regime, Parkinson volatility, and fractional-diff close. The v2 path also records feature-family and bar-clock diagnostics. Historical side-data joins remain gated by [#8](https://github.com/pranay5255/ETHPredict/issues/8) and feature validity by [#9](https://github.com/pranay5255/ETHPredict/issues/9).
- The older feature-preprocessor path targets `close` and `volume`; v2 research uses next-5-minute and next-hour return/direction targets.

### V2 Artifact-Backed Research Pipeline

Status: implemented as an exploratory research path, with open validity and reporting issues.

- The v2 runner builds multi-horizon data, out-of-fold base predictions, triple-barrier meta candidates, a meta-labeler, directional alpha backtests, and per-trial manifests. It separates validation and test artifacts but still needs the exact-span validation and stronger backtest evidence in [#15](https://github.com/pranay5255/ETHPredict/issues/15) and [#21](https://github.com/pranay5255/ETHPredict/issues/21).
- [#14](https://github.com/pranay5255/ETHPredict/issues/14) is closed: label spans and uniqueness sample weights are implemented. [#30](https://github.com/pranay5255/ETHPredict/issues/30) remains open to compare weighting choices.
- Experiment issues #25–#29 are closed gates, documented in `context/afml_freeze_regen_20260802_report.md` and linked individually in `TASKS.md`. The #27–#29 variants generated no validation or test trades under the frozen threshold, so they establish instrumentation, not tradable alpha. Parent issues [#23](https://github.com/pranay5255/ETHPredict/issues/23), [#24](https://github.com/pranay5255/ETHPredict/issues/24), [#8](https://github.com/pranay5255/ETHPredict/issues/8), [#9](https://github.com/pranay5255/ETHPredict/issues/9), and [#10](https://github.com/pranay5255/ETHPredict/issues/10) remain open.
- Local Trackio contains prior v2 runs. Per-checkpoint identity, out-of-fold forecast metrics, delivery receipts, and local readback checks are being hardened under #23; the full issue remains open.

### Lighter Compare Experiments

Status: smoke-tested exploratory comparison, not the primary path for future checkpoint promotion.

- `configs/lighter_experiments.yml` defines two active targets: `triple_barrier` and `next_hour_return`.
- `src.experiments.lighter_compare` runs the `PriceLSTM`, `MetaMLP`, and `ConfidenceGRU` stack on `cuda:0`.
- The runner includes one forward/backward stack smoke, one or more neural trials per target, CPU ARIMA/SARIMAX baselines, and GLFT backtest metrics for finalist ranking.
- Smoke mode verifies plumbing on a small tail slice; small full-data staged trials have also been run on the active 5m file.
- ARIMA and SARIMAX baselines remain CPU-bound.

### Configuration

Status: active schema and loader simplified.

- `configs/config.yml` is Lighter-only for the top-level pipeline.
- `configs/lighter_experiments.yml` remains available for exploratory comparisons; `configs/config.yml` is the primary v2 research config.
- `configs/schema.yaml` no longer models active `bribe` or `sim` sections.
- `src/config/loader.py` validates the active config and exposes a compact typed config object.
- `src/config/__init__.py` now exports the active config loader symbols instead of non-existent sweep/core modules.

### Archive

Status: complete for this refactor.

- Legacy CSVs were moved into `archive/legacy_data_sources/data`.
- Legacy multi-source code/config/docs were copied into `archive/legacy_data_sources/code` and `archive/legacy_data_sources/docs`.
- Active loaders do not read from the archive.

### Tests and Smoke Checks

Status: full suite and GPU smoke checks were previously reported passing; the current Trackio changes have focused tests and a local delivery/readback smoke check.

Previously verified commands:

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

Status: the legacy top-level runner still needs verification after the refactor; it is not the primary v2 research path.

- `runner.py` now initializes the Lighter-only preprocessor.
- Backtesting still has fallback behavior using shifted prices if model prediction integration fails.
- Next step: run `python runner.py configs/config.yml` and fix any integration issues from model/training/backtest boundaries.

### Model Prediction Integration in Backtesting

Status: partial in the legacy runner. The v2 path already backtests saved model predictions, but its research evidence remains incomplete.

- The code attempts model prediction when possible.
- The fallback shifted-price path still exists.
- Next step: make model inference the primary path and fail clearly if model outputs cannot be produced.

### Full Lighter Compare Experiment

Status: smoke-tested; a full run remains optional exploratory work.

- Smoke mode has verified the end-to-end experiment plumbing on `cuda:0`.
- Future finalist review should prioritize prediction evidence; GLFT results remain exploratory execution research.

### Small Staged 5m Trials

Status: completed as exploratory full-data experiments.

- Added small full-data staged configs for `next_hour_return` and `triple_barrier` on 5m data.
- Each run used 26,208 sequence samples, 8 model trials, and 6 GLFT strategy trials.
- The initial parallel launch showed that two full-data jobs can overrun GPU memory during full-dataset base-prediction generation; serial fallback completed cleanly.
- In the completed runs, `next_hour_return` had better held-out strategy PnL, while `triple_barrier` had stronger validation hit ratio but weaker test transfer.
- GLFT results are exploratory only because the current simulator is stylized and random-fill based.

### Hierarchical Meta-Labeling Roadmap

Status: v2 MVP implemented; model definitions, calibration, and validation remain open research work.

- V2 predicts next-5-minute and next-hour returns and directions; [#11](https://github.com/pranay5255/ETHPredict/issues/11) covers calibration and candidate mapping diagnostics.
- The v2 runner generates out-of-fold base predictions with fixed-gap purged walk-forward CV; exact label-span purging and CPCV remain [#15](https://github.com/pranay5255/ETHPredict/issues/15).
- V2 constructs triple-barrier meta labels from proposed long/short signals after configured costs; [#13](https://github.com/pranay5255/ETHPredict/issues/13) audits barrier geometry and ambiguous price paths.
- The v2 meta-labeler uses signal-success candidates; the older `ConfidenceGRU` path remains exploratory.
- V2 trains a meta model from base predictions and candidate context; [#18](https://github.com/pranay5255/ETHPredict/issues/18) covers calibration, model comparison, and threshold selection.

### Backtesting and GLFT Positioning

Status: v2 has a directional alpha backtest for debugging; stronger cost, sizing, and statistical evidence remains open.

- The v2 path reports directional alpha validation/test results. [#19](https://github.com/pranay5255/ETHPredict/issues/19), [#20](https://github.com/pranay5255/ETHPredict/issues/20), and [#21](https://github.com/pranay5255/ETHPredict/issues/21) cover sizing, execution-cost realism, and backtest statistics.
- GLFT should move to an optional execution layer after alpha validation: the policy sets target inventory or reservation-price skew, and GLFT handles passive quote placement.

### Lighter Side-Data Features

Status: collected but not integrated.

- Funding, mark price, order book, recent trades, and exchange metrics are available under `data/lighter/`.
- Active features intentionally ignore side data for now.
- Recommended first additions: mark/trade basis, funding cost features, funding z-scores, and time-to-next-funding.
- Later additions: historical order-book imbalance/microprice, trade-flow imbalance/VWAP, open-interest changes, RSI, MACD, Bollinger bands, and volume profile.

### Current Data, Model Freeze, and Dashboard

Status: not implemented.

- [#43](https://github.com/pranay5255/ETHPredict/issues/43) specifies five-minute public Lighter data and feature snapshots with as-of timestamps, coverage, and stale states. The current collector remains date-bounded and latest side-data files are overwritten.
- [#44](https://github.com/pranay5255/ETHPredict/issues/44) specifies explicit promotion of a complete, reviewed v2 checkpoint and read-only current inference. Current base state-dict artifacts are not yet complete deployable base-plus-meta packages with preprocessing state.
- [#45](https://github.com/pranay5255/ETHPredict/issues/45) specifies a local dashboard with current features and model outputs, per-checkpoint performance linked to Trackio, run stages with duration/resource/failure metrics, AFML phases, and simulated backtest evidence. No trade recommendation or signed order submission is part of this view.

### Training Environment

Status: GPU-first policy implemented for neural experiments.

- The Lighter feature preprocessor remains Lighter/OHLCV scoped.
- Model/training modules depend on the uv-managed Torch environment.
- Neural training fails fast without CUDA unless CPU use is explicitly allowed.
- A future policy decision is still needed on whether CPU neural fallback should be documented beyond tests/development.

### Documentation

Status: task and status documents reconciled with GitHub issue states on 2026-09-23.

- README, TASKS, PROJECT_STATUS, CHANGELOG, and Lighter context docs describe the active Lighter-only path.
- README, TASKS, PROJECT_STATUS, and CHANGELOG describe the uv CUDA 12.8 experiment path. TASKS and this file now identify v2 as the primary future research path and link the new decision-support issues.
- Older broad research docs remain in `context/` as planning references unless explicitly archived later.

## Not Active in This Phase

- Binance, DeFiLlama, and Santiment feature joins.
- DEX simulation.
- Bribe/MEV optimization.
- Bayesian optimization; v2 grid search exists for research trials.
- Signed Lighter trading through the decision-support dashboard or live execution. Separate guarded testnet scripts exist in this repository.
- Production deployment service.

## Completion Snapshot

| Area | Status |
| --- | --- |
| Lighter data collection | Complete for raw pass |
| Raw OHLCV validation | Complete for current file |
| Lighter-only feature generation | Active and smoke-tested |
| uv CUDA 12.8 dependency path | Active and verified |
| CUDA 4090 Torch validation | Passing |
| Lighter compare smoke runner | Passing as exploratory path |
| V2 multi-horizon and meta-label pipeline | Implemented, research-grade validation pending |
| Label-span uniqueness weights (#14) | Implemented; comparison #30 pending |
| Trackio local project | Existing runs; checkpoint and delivery hardening under #23 |
| Legacy source archive | Complete |
| Config simplification | Complete |
| Full uv test suite | Previously reported passing; current changes checked with focused tests |
| End-to-end runner | Needs post-refactor verification |
| Model-backed backtest predictions | V2 active; legacy runner partial |
| Full Lighter compare experiment | Pending |
| Small staged 5m experiments | Complete, exploratory |
| Meta-labeling with fixed-gap purged CV | V2 MVP implemented; exact-span purging pending |
| Directional alpha backtest | Debugging path implemented; stronger evidence pending |
| Lighter side-data joins | Not started |
| Refreshed data and feature snapshots (#43) | Not started |
| Manually frozen current inference (#44) | Not started |
| Local decision-support dashboard (#45) | Not started |
| Live trading/execution in dashboard | Out of scope |

## Next Recommended Steps

1. Finish [#23](https://github.com/pranay5255/ETHPredict/issues/23) and [#24](https://github.com/pranay5255/ETHPredict/issues/24) so every v2 checkpoint and trial has reliable accounting and immutable identity.
2. Build the public current-data snapshots in [#43](https://github.com/pranay5255/ETHPredict/issues/43), then the explicit model freeze and inference path in [#44](https://github.com/pranay5255/ETHPredict/issues/44).
3. Build the local dashboard in [#45](https://github.com/pranay5255/ETHPredict/issues/45), showing available evidence and clear gaps until [#17](https://github.com/pranay5255/ETHPredict/issues/17) and [#21](https://github.com/pranay5255/ETHPredict/issues/21) are complete.
4. Continue the AFML issue dependencies in `TASKS.md`; use purged validation for research choices and keep simulated backtests distinct from current model outputs.
