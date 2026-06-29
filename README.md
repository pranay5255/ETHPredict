# ETHPredict

ETHPredict is currently scoped to public Lighter ETH perpetual market data. The active pipeline collects Lighter OHLCV candles, builds OHLCV-only features, and keeps legacy non-Lighter work recoverable under `archive/legacy_data_sources`.

The current experiment path uses `uv` for dependency management and a CUDA 12.8 PyTorch environment on `cuda:0` for neural training. ARIMA and SARIMAX baselines remain CPU-bound.

## Active Scope

- **Data source**: Lighter public ETH perp candles, `market_id: 0`.
- **Feature source**: OHLCV only from `data/raw/*-lighter-*.csv`.
- **Experiment runner**: `src.experiments.lighter_compare` for Lighter-only neural trials, ARIMA/SARIMAX baselines, and GLFT metric ranking.
- **GPU policy**: neural smoke tests and training default to `cuda:0`; CPU neural runs require explicit `--allow-cpu`.
- **Side data**: Lighter mark-price candles, 1h fundings, latest funding rates, order book, recent trades, and exchange metrics are collected under `data/lighter/` but are not joined into model features yet.
- **Archived**: Binance, DeFiLlama, Santiment, DEX simulation, bribe optimization, and parameter-optimization config/code are archived under `archive/legacy_data_sources`.

## Project Layout

```text
configs/
  config.yml                  # Active Lighter-only pipeline config
  lighter_experiments.yml     # Lighter compare experiment config
  staged_trial_smoke.yml      # Staged model-trial/backtest artifact smoke config
  staged_trial_small_*.yml    # Small full-data staged experiments for 5m research
  lighter_trading.yml         # Non-secret Lighter testnet trading defaults
  schema.yaml                 # Active config validation schema
scripts/
  lighter_collect_data.py     # Public Lighter data collector
  lighter_trading_setup.py    # Lighter testnet account/API key checks
  lighter_place_test_order.py # Guarded testnet order submit/cancel helper
  lighter_subaccount_ops.py   # Dry-run-first subaccount create/fund helpers
  lighter_export_example_config.py # Export .env creds for upstream examples
src/
  trading/                    # Signed Lighter trading config helpers
  experiments/lighter_compare.py # Lighter neural/baseline experiment runner
  experiments/staged_trial.py # Stage 0/1/2 artifact-backed trial orchestration
  data/features_all.py        # Lighter-only OHLCV feature generation
  data/lighter_client.py      # Public Lighter REST client
  config/loader.py            # Active config loader
  csv_loader.py               # CSV schema validation helper
  models/                     # Existing model definitions
  training/                   # Device policy and training utilities
  market_maker/               # GLFT quote and inventory logic
  simulator/                  # Price-data backtest framework
archive/legacy_data_sources/  # Recoverable legacy source/config/code archive
data/raw/                     # Active model OHLCV exports
data/lighter/                 # Lighter side-data exports
external/lighter-python/      # Official Lighter SDK examples submodule
```

## Setup

Use `uv` for the active GPU experiment path. `pyproject.toml` and `uv.lock` are the source of truth for current experiments, including the official PyTorch CUDA 12.8 wheel index.

```bash
uv sync
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected GPU validation on the 4090 machine:

```text
2.11.0+cu128
True
NVIDIA GeForce RTX 4090
```

The legacy `requirements.txt` flow remains available for older setup work, but it is no longer the active dependency source for GPU experiments:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Collect Lighter Data

Use the defaults in `configs/config.yml`:

```bash
python scripts/lighter_collect_data.py --config configs/config.yml
```

Expected active raw output:

```text
data/raw/ETHUSDT-5m-lighter-20260328-20260628.csv
```

The collector also writes Lighter-native side data under `data/lighter/`; those files are retained for audit and future research, not joined into the current OHLCV feature set.

## Lighter Testnet Trading

Signed trading uses the official Lighter SDK on the testnet endpoint:

```text
https://testnet.zklighter.elliot.ai
```

Non-secret defaults are in `configs/lighter_trading.yml`; local secrets are in
`.env`, which is gitignored. Required for normal trading:

```text
LIGHTER_ACCOUNT_INDEX
LIGHTER_API_KEY_INDEX
LIGHTER_API_PRIVATE_KEY
```

Required only to create or rotate a Lighter API key:

```text
LIGHTER_ETH_PRIVATE_KEY
```

Inspect the expected environment:

```bash
uv run python scripts/lighter_trading_setup.py env
```

After funding/creating a testnet account and adding credentials to `.env`, run
a read-only setup check:

```bash
uv run python scripts/lighter_trading_setup.py check
```

To create and register a Lighter API key from your testnet L1 wallet key:

```bash
uv run python scripts/lighter_trading_setup.py register-api-key --api-key-index 3
```

Prepare a dry-run ETH perp order:

```bash
uv run python scripts/lighter_place_test_order.py --side sell --size-base 0.0050 --price 4050.00
```

Submit the order to testnet and cancel it immediately:

```bash
uv run python scripts/lighter_place_test_order.py --side sell --size-base 0.0050 --price 4050.00 --submit
```

The ETH perp defaults match Lighter examples and current metadata: `market_index: 0`,
`size_decimals: 4`, and `price_decimals: 2`.

### Upstream SDK Examples

The official SDK is checked out as a submodule at `external/lighter-python`. After
cloning this project elsewhere, initialize it with:

```bash
git submodule update --init --recursive
```

The most useful upstream examples are:

- `external/lighter-python/examples/get_info.py`: account and market read examples on testnet. It also calls `apikeys` for a hardcoded account/key, so it may fail if that key does not exist; use this repo's `public` command for the first smoke test.
- `external/lighter-python/examples/system_setup.py`: creates Lighter API keys from an L1 wallet key and writes `api_key_config.json`. This repo uses `scripts/lighter_trading_setup.py register-api-key` instead, so secrets stay in `.env`.
- `external/lighter-python/examples/create_modify_cancel_order_http.py`: create, modify, and cancel a limit order via HTTP `sendTx`.
- `external/lighter-python/examples/create_modify_cancel_order_ws.py`: same transaction flow, sent over websocket.
- `external/lighter-python/examples/read-only-auth/`: dedicated API key `253` and pre-generated auth tokens for read-only account queries. Use this later for services that should not hold API private keys at runtime.
- `external/lighter-python/examples/paper_trading_*.py`: simulated trading against order book data; no API keys required.

To run upstream trading examples from this repo root after `.env` is populated:

```bash
uv run python scripts/lighter_export_example_config.py
uv run python external/lighter-python/examples/create_modify_cancel_order_http.py
```

`api_key_config.json` is ignored by git. The upstream order example uses `0.1 ETH`; start with this repo's smaller cancel-after-submit helper first.

### Auth-Gated Testing Plan

1. Public endpoint smoke test, no secrets:

```bash
uv run python scripts/lighter_trading_setup.py public
```

2. Confirm this repo sees the expected env vars:

```bash
uv run python scripts/lighter_trading_setup.py env
```

3. In the Lighter testnet app, connect the wallet and fund the account from the faucet. Then set `LIGHTER_ETH_PRIVATE_KEY` in `.env` only long enough to register an API key:

```bash
uv run python scripts/lighter_trading_setup.py register-api-key --api-key-index 3
```

This writes `LIGHTER_ACCOUNT_INDEX`, `LIGHTER_API_KEY_INDEX`, and `LIGHTER_API_PRIVATE_KEY` back to `.env`. Remove `LIGHTER_ETH_PRIVATE_KEY` after registration if you do not need more key rotation.

4. Run the read-only authenticated check:

```bash
uv run python scripts/lighter_trading_setup.py check
```

This verifies account lookup, next nonce, API key matching, auth token creation, and `accountActiveOrders` without placing an order.

5. Dry-run a tiny ETH perp order and inspect integer scaling:

```bash
uv run python scripts/lighter_place_test_order.py --side sell --size-base 0.0050 --price 4050.00
```

Expected scaling is `base_amount=50` and `price_int=405000`.

6. Submit the same tiny order to testnet and cancel it immediately:

```bash
uv run python scripts/lighter_place_test_order.py --side sell --size-base 0.0050 --price 4050.00 --submit
```

This is the first write test. It exercises signed create-order `sendTx`, nonce handling, and signed cancel-order `sendTx`.

7. Only after steps 1-6 pass, run the upstream HTTP then websocket examples:

```bash
uv run python scripts/lighter_export_example_config.py
uv run python external/lighter-python/examples/create_modify_cancel_order_http.py
uv run python external/lighter-python/examples/create_modify_cancel_order_ws.py
```

8. Defer market orders, margin/leverage changes, transfers, withdrawals, public pools, and spot examples until the limit-order create/cancel path is boringly repeatable on testnet.

### Safe Subaccount Model

Use two separate local env files and never put the L1 wallet private key in the bot env:

- Admin funding env: source account API key only. Use it for one-time subaccount creation and USDC transfers. Example path: `.env.lighter.admin.testnet`.
- Bot subaccount env: subaccount account index and subaccount API key only. Use it for trading, experiments, and live/paper execution checks. Example path: `.env.lighter.testnet`.

The L1 wallet private key is only needed to register or rotate an API key on an account. For mainnet, do that outside this repo or in a short-lived shell, then copy only the resulting subaccount API key into the bot env.

Create a subaccount from an admin/source account, dry-run first:

```bash
uv run python scripts/lighter_subaccount_ops.py --env-file .env.lighter.admin.testnet create-subaccount
uv run python scripts/lighter_subaccount_ops.py --env-file .env.lighter.admin.testnet create-subaccount --submit
```

After the subaccount exists, find its account index without using any private key:

```bash
uv run python scripts/lighter_subaccount_ops.py --env-file .env.lighter.admin.testnet list-l1 --l1-address <YOUR_L1_ADDRESS>
```

Then register an API key for that subaccount. Do not use the master account index here; set `LIGHTER_ACCOUNT_INDEX` to the subaccount index before registration:

```bash
uv run python scripts/lighter_trading_setup.py --env-file .env.lighter.testnet register-api-key --api-key-index 3
uv run python scripts/lighter_trading_setup.py --env-file .env.lighter.testnet check
```

Fund the subaccount with USDC from the admin/source account, dry-run first:

```bash
uv run python scripts/lighter_subaccount_ops.py --env-file .env.lighter.admin.testnet transfer-usdc --to-account-index <SUBACCOUNT_INDEX> --amount 100
uv run python scripts/lighter_subaccount_ops.py --env-file .env.lighter.admin.testnet transfer-usdc --to-account-index <SUBACCOUNT_INDEX> --amount 100 --submit
```

Check the funded subaccount and run trading commands only with the bot env:

```bash
uv run python scripts/lighter_subaccount_ops.py --env-file .env.lighter.testnet account
uv run python scripts/lighter_place_test_order.py --env-file .env.lighter.testnet --side sell --size-base 0.0050 --price 4050.00
```

Mainnet must be deliberately separate: use `.env.lighter.mainnet`, set `LIGHTER_NETWORK=mainnet`, set `LIGHTER_BASE_URL=https://mainnet.zklighter.elliot.ai`, and leave `LIGHTER_ALLOW_MAINNET=false` until you are intentionally ready. Write helpers also require `--confirm-mainnet`, so accidental mainnet writes fail even if the env points at mainnet.

## Build Features

Build sequence features from the active Lighter raw CSV:

```bash
python -m src.data.features_all full_features --data-dir data --granularity 5m --sequence-length 288 --out-dir /tmp/ethpredict-lighter-features
```

`DataPreprocessor` intentionally loads only files matching:

```text
data/raw/ETHUSDT-<resolution>-lighter-*.csv
```

Legacy-looking files such as `ETHUSDT-5m-2025-04.csv` are ignored by active loaders.

## Run Pipeline

The top-level runner takes a config path:

```bash
python runner.py configs/config.yml
```

The runner currently uses the Lighter-only feature stack, existing model/training components, GLFT market-making setup, and price-data backtesting. Full model-prediction integration in backtesting remains a follow-up item.

## Run Lighter Experiments

The uv-managed experiment runner compares the three-model neural stack against CPU ARIMA/SARIMAX baselines for both active targets:

- `triple_barrier`
- `next_hour_return`

Smoke run:

```bash
uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml --smoke
```

Full configured run:

```bash
uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml
```

Neural training defaults to `cuda:0` and fails fast if CUDA is unavailable. CPU neural runs require an explicit override:

```bash
uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml --smoke --device cpu --allow-cpu
```

ARIMA and SARIMAX baselines remain CPU-bound.

## Run Staged Trials

Use the staged runner when you want Stage 1 model/hyperparameter trials to produce frozen artifacts and Stage 2 backtests to consume the selected Stage 1 predictions. The starter smoke config runs one model trial and a small GLFT strategy grid; expand `staged_trial.stage1.search_space` or `staged_trial.stage2.strategy_search` for longer searches.

```bash
uv run python -m src.experiments.staged_trial --config configs/staged_trial_smoke.yml --smoke
```

CPU smoke runs are explicit:

```bash
uv run python -m src.experiments.staged_trial --config configs/staged_trial_smoke.yml --smoke --device cpu --allow-cpu
```

Each run writes a single artifact directory under `staged_trial.artifact_root`:

```text
artifacts/runs/<run_id>/
  resolved_config.yml
  environment.json
  stage0_features/manifest.json
  stage1_model/best_model_manifest.json
  stage1_model/trials/<trial_id>/predictions_test.parquet
  stage2_backtest/best_strategy_manifest.json
  stage2_backtest/strategies/<strategy_id>/{trades,timeseries,pnl}.parquet
  report/report.json
  report/report.md
```

Stage 2 intentionally consumes the best Stage 1 `predictions_test.parquet` and model manifest instead of silently rebuilding model state.


## Trackio Dashboard

Trackio logging is enabled in `configs/config.yml` for the active v2 research path and writes to the local Trackio store by default. Launch the local dashboard with the Python entry point when CLI networking is awkward in sandboxed shells:

```bash
uv run python -c "import trackio; trackio.show(project='ethpredict', open_browser=False, host='127.0.0.1', server_port=7860)"
```

The CLI alternative is also available:

```bash
uv run trackio show --project ethpredict
```


## Research Roadmap

The next research direction is a hierarchical signal stack rather than a single model plus GLFT backtest. The staged runner should evolve into these layers:

1. **Level 0 multi-horizon forecaster**: use the same sequence encoder to predict `next_5m_return`, `next_5m_direction`, `next_hour_return`, `next_hour_direction`, and volatility/uncertainty. The 5m horizon is useful for execution timing; the 1h horizon is useful for slower directional edge.
2. **Level 1 signal proposer**: convert base forecasts into candidate long/short/no-trade signals. A signal is proposed only when predicted edge exceeds estimated spread, fees, slippage, and funding cost.
3. **Level 2 true meta-labeler**: train a classifier to decide whether to take the proposed base signal. The current `ConfidenceGRU` predicts whether `y_dir != 0`, which is an event-confidence label, not true meta-labeling.
4. **Level 3 policy/risk layer**: choose horizon, side, size, stop/take-profit, and no-trade decisions from expected net edge, meta probability, volatility, inventory, and drawdown constraints.
5. **Execution layer**: use simple taker/maker execution in research backtests first. Keep GLFT as a later passive quoting/execution module, not the primary alpha evaluator.

### Triple-Barrier Meta-Labeling

For true meta-labeling, first train the base forecaster and generate **out-of-sample** predictions with purged walk-forward cross-validation. Then build labels from proposed trades:

- `side`: `+1` for proposed long, `-1` for proposed short.
- `entry_price`: close, mark, or executable mid at signal time.
- `vertical_barrier`: `1` bar for `next_5m` or `12` bars for `next_hour` on 5m data.
- `profit_taking_barrier`: volatility-scaled expected upside after costs, for example `kappa_profit * realized_vol`.
- `stop_loss_barrier`: volatility-scaled downside after costs, for example `kappa_stop * realized_vol`.
- `meta_label`: `1` if the side-adjusted path hits profit-taking before stop-loss, or exits positive after all modeled costs at the vertical barrier; otherwise `0`.

The meta model inputs should include base predictions, direction probabilities, horizon disagreement, volatility/regime features, funding/cost estimates, and eventually order-book/trade-flow features. Samples without a base signal should be excluded from binary meta-label training or handled by a separate no-trade policy label.

### Purged Cross-Validation

Meta labels must be trained from out-of-sample base predictions. Use purged walk-forward CV:

- Split history into chronological folds.
- For each fold, train the base model on earlier data only.
- Purge overlapping lookahead windows around validation labels; use at least the maximum label horizon, currently `288` bars for 24h triple-barrier timeout.
- Apply an embargo after each validation fold before later training data is allowed.
- Store fold predictions as artifacts and train the meta-labeler only on these out-of-sample predictions.
- Reserve the final test split for one untouched evaluation after thresholds and policy settings are chosen on validation.

### Unused Feature Sources and Fit Points

The active model already uses OHLCV-derived returns, volatility, entropy, CUSUM/SADF flags, volatility regime, Parkinson volatility, and fractional-diff close. README/config ideas that are not yet integrated should enter the pipeline in Stage 0:

- **Mark-price candles**: mark/trade basis, rolling premium, premium z-score.
- **Funding rates**: current funding, funding z-score, time-to-next-funding, expected holding cost.
- **Order-book snapshots**: spread, depth imbalance, microprice, liquidity slope, top-of-book pressure. These require historical snapshots before they should be trusted for training.
- **Recent trades**: aggressive flow imbalance, rolling VWAP, trade count, realized slippage.
- **Exchange metrics**: open interest, OI change, volume/OI ratio, crowding/leverage proxies.
- **Technical indicators**: RSI, MACD, Bollinger bands, and volume profile as ordinary Stage 0 features.
- **Tree base models**: XGBoost/LightGBM/CatBoost can be added as tabular Level 0 models using latest-bar and rolling-window summary features, then ensembled with neural outputs before meta-labeling.

### Backtesting Design

GLFT is useful as a plumbing and future passive-execution component, but the main research evaluator should be a directional alpha backtest first:

- Select thresholds, horizon choice, sizing, stops, and take-profits on validation only.
- Evaluate once on the untouched test split.
- Model spread, taker/maker fees, slippage, funding, latency, position limits, and max drawdown.
- Report gross PnL, net PnL, fees, turnover, exposure, hit rate, average win/loss, drawdown, and coverage.
- Add GLFT later by letting the policy set target inventory or reservation-price skew while GLFT decides passive quote placement.

Recommended build order:

1. Add `next_5m_return` and multi-horizon prediction outputs.
2. Generate purged walk-forward base predictions.
3. Build triple-barrier meta labels from net profitability of proposed signals.
4. Train and evaluate a true meta-label classifier with thresholded coverage/performance reports.
5. Replace Stage 2 with a validation-selected directional alpha backtest.
6. Reintroduce GLFT as an optional passive execution layer once alpha quality is measurable.

## Configuration

The active pipeline config keeps only the sections needed for this phase:

```yaml
experiment:
  id: exp_1
  seed: 42
  trials: 1000

data:
  sources:
    - lighter
  lighter:
    base_url: "https://mainnet.zklighter.elliot.ai"
    symbol: "ETH"
    market_type: "perp"
    market_id: 0
    pipeline_symbol: "ETHUSDT"
    resolution: "5m"
    funding_resolution: "1h"
    start_date: "2026-03-28"
    end_date: "2026-06-28"
    count_back: 500
    request_pause_seconds: 0.1

features:
  frac_diff_order: 0.5
  include: [vol_adj_flow, rsi, macd, bollinger, volume_profile]

model:
  type: hierarchical
  level0:
    algo: xgboost
    params:
      max_depth: 8
      eta: 0.1
      tree_method: gpu_hist

market_maker:
  strategy: glft
  gamma: 0.5
  inventory_limit: 10000
  quote_spread: 0.001

backtest:
  start: "2024-01-01"
  end: "2025-01-01"
  initial_capital: 100000
```

DEX simulation, bribe/MEV optimization, and parameter sweep config were removed from the active config and archived.

The active experiment config is separate:

```yaml
data:
  granularity: 5m

targets:
  - triple_barrier
  - next_hour_return

labels:
  timeout: 288
  volatility_window: 288
  next_hour_horizon: 12

training:
  sequence_length: 288
  hidden_size: 16
  num_layers: 1
  batch_size: 16
  epochs: 3
  neural_trials_per_target: 2

smoke:
  max_rows: 640
  neural_trials_per_target: 1
  epochs: 1
```

## Verification

Current uv/GPU checks:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
uv run pytest
uv run pytest tests/test_lighter_experiments.py
uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml --smoke
uv lock --check
```

Focused checks from the Lighter-only refactor remain useful:

```bash
python -m pytest tests/test_lighter_client.py tests/test_lighter_preprocessor.py tests/test_config.py tests/test_runner.py
python -m py_compile src/data/features_all.py src/data/lighter_client.py scripts/lighter_collect_data.py src/config/loader.py src/config/__init__.py src/main.py runner.py
python -m src.data.features_all full_features --data-dir data --granularity 5m --sequence-length 288 --out-dir /tmp/ethpredict-lighter-features
```

## Notes

- The active raw Lighter file collected for this pass has 26,496 5m rows from `2026-03-28 00:00:00 UTC` through `2026-06-27 23:55:00 UTC`; 24-hour model lookbacks use 288 bars and the next-hour target uses 12 bars.
- Optional Lighter `exchangeMetrics` outputs are best-effort daily/all-period side data. Mark-price 5m, funding 1h, order-book snapshots, recent-trade snapshots, and exchange metrics stay under `data/lighter/` and are not joined into OHLCV training features.
- `requirements.txt` is preserved for legacy setup until a later cleanup; uv metadata is the active experiment dependency source.
- `archive/legacy_data_sources/README.md` lists archived files and why they are no longer active.
