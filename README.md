# ETHPredict

ETHPredict is currently scoped to public Lighter ETH perpetual market data. The active pipeline collects Lighter OHLCV candles, builds OHLCV-only features, and keeps legacy non-Lighter work recoverable under `archive/legacy_data_sources`.

The current experiment path uses `uv` for dependency management and a CUDA 12.8 PyTorch environment on `cuda:0` for neural training. ARIMA and SARIMAX baselines remain CPU-bound.

## Active Scope

- **Data source**: Lighter public ETH perp candles, `market_id: 0`.
- **Feature source**: OHLCV only from `data/raw/*-lighter-*.csv`.
- **Experiment runner**: `src.experiments.lighter_compare` for Lighter-only neural trials, ARIMA/SARIMAX baselines, and GLFT metric ranking.
- **GPU policy**: neural smoke tests and training default to `cuda:0`; CPU neural runs require explicit `--allow-cpu`.
- **Side data**: Lighter funding, mark price, order book, and recent trades are collected under `data/lighter/` but are not joined into features yet.
- **Archived**: Binance, DeFiLlama, Santiment, DEX simulation, bribe optimization, and parameter-optimization config/code are archived under `archive/legacy_data_sources`.

## Project Layout

```text
configs/
  config.yml                  # Active Lighter-only pipeline config
  lighter_experiments.yml     # Lighter compare experiment config
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
data/raw/ETHUSDT-1h-lighter-20260328-20260628.csv
```

The collector also writes Lighter-native side data under `data/lighter/`.

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
python -m src.data.features_all full_features --data-dir data --sequence-length 24 --out-dir /tmp/ethpredict-lighter-features
```

`DataPreprocessor` intentionally loads only files matching:

```text
data/raw/ETHUSDT-<resolution>-lighter-*.csv
```

Legacy-looking files such as `ETHUSDT-1h-2025-04.csv` are ignored by active loaders.

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
    resolution: "1h"
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
targets:
  - triple_barrier
  - next_hour_return

training:
  sequence_length: 24
  hidden_size: 16
  num_layers: 1
  batch_size: 16
  epochs: 3
  neural_trials_per_target: 2

smoke:
  max_rows: 128
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
python -m src.data.features_all full_features --data-dir data --sequence-length 24 --out-dir /tmp/ethpredict-lighter-features
```

## Notes

- The active raw Lighter file collected for this pass has 2,208 hourly rows from `2026-03-28 00:00:00 UTC` through `2026-06-27 23:00:00 UTC`.
- Optional Lighter `exchangeMetrics` outputs are best-effort and may be skipped when the API rejects the period/filter combination.
- `requirements.txt` is preserved for legacy setup until a later cleanup; uv metadata is the active experiment dependency source.
- `archive/legacy_data_sources/README.md` lists archived files and why they are no longer active.
