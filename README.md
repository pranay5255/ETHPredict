# ETHPredict

ETHPredict researches directional signals using public Lighter ETH perpetual data. The default feature path uses ETH OHLCV CSVs; optional historical side-data groups are coverage gated. A separate five-market mainnet Parquet lake is now collected. Legacy non-Lighter work remains under `archive/legacy_data_sources`.

The current experiment path uses `uv` for dependency management and a CUDA 12.8 PyTorch environment on `cuda:0` for neural training. ARIMA and SARIMAX baselines remain CPU-bound.

## Current status — 2026-10-27 merge review

- **History available:** ETH/BTC/SOL/HYPE/NEAR, June 4, 2025 through a partial
  October 16, 2026; 25 readable Parquet datasets, 425 verified partition hashes,
  303.10 MB (289.06 MiB). Mark history begins August 25, 2025.
- **Data caveat:** funding has 31 shared missing hours per market. Six missing ETH
  hours exist in an older CSV; exchange-wide absence is unproven. Recovery is
  [#54](https://github.com/pranay5255/ETHPredict/issues/54).
- **Integration:** training still reads CSVs; the lake does not change old runs.
  See [the data lake runbook and audit](context/lighter_data_lake.md).
- **Code:** [PR #51](https://github.com/pranay5255/ETHPredict/pull/51) adds
  #16/#17/#20 and includes [PR #53](https://github.com/pranay5255/ETHPredict/pull/53),
  previously merged into its branch (forecast repair/opportunity study/economic gate).
  See [the merge audit](context/pr51_merge_review_20261027.md).
- **Economics:** the first bounded final test lost 136.174 net on 103 trades.
  The subsequent development study rejected every configuration and used no new
  final test. No strategy is promoted and no profitability probability is established.
- **Queue after #51:** 18 open issues (6 implementation/data, 12 experiments); historical
  backfill alone completes none of their acceptance criteria.

Start with [PROJECT_STATUS.md](PROJECT_STATUS.md), [TASKS.md](TASKS.md),
[experiment-readiness.md](experiment-readiness.md) and
[the visual implementation roadmap](ethpredict-implementation-roadmap.html).

## Active Scope

- **Research market**: Lighter public ETH perp candles, `market_id: 0`; the separate lake also contains BTC, SOL, HYPE and NEAR.
- **Feature source**: OHLCV only from `data/raw/*-lighter-*.csv`.
- **Experiment runner**: `src.experiments.lighter_compare` for Lighter-only neural trials, ARIMA/SARIMAX baselines, and GLFT metric ranking.
- **GPU policy**: neural smoke tests and training default to `cuda:0`; CPU neural runs require explicit `--allow-cpu`.
- **Side data**: Lighter mark-price candles, 1h fundings, latest funding rates, order book, recent trades, and exchange metrics are collected under `data/lighter/`. `src/data/side_data.py` can join config-selected historical groups with per-split coverage gates (`data.side_data.groups`), but no group is enabled in `configs/config.yml`, so default runs use OHLCV features only.
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
  lighter_collect_data.py     # Public Lighter CSV collector
  lighter_backfill_lake.py    # Multi-market monthly Parquet history
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
data/lake/lighter/            # Ignored five-market monthly Parquet lake
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

The collector also writes Lighter-native side data under `data/lighter/`; those files are retained for audit and as inputs to optional side-data groups. Only historical series (for example 1h fundings and mark-price candles) can feed a group; latest order-book and recent-trade snapshots cannot.

### Multi-market history lake

`scripts/lighter_backfill_lake.py` backfills full public history for several perps into the git-ignored `data/lake/lighter/` (parquet, one file per month):

```bash
python scripts/lighter_backfill_lake.py --symbols ETH BTC SOL HYPE NEAR
```

Each market gets `candles_1m`, `candles_5m`, `candles_1h`, `mark_price_5m`, and `fundings_1h`, plus `_manifest.json` with row counts, missing intervals, and partition hashes. Reruns skip existing closed-month files and refresh the current UTC month; they do not recover gaps inside old files. Every partition shares one schema, so `pd.read_parquet("data/lake/lighter/ETH/candles_1m")` loads a whole dataset; `--repair --symbols ...` rewrites partitions saved by older versions of the script (no network).

`--env-file .env.testnet` reads only `LIGHTER_BASE_URL` and `LIGHTER_NETWORK` (API keys are ignored) and writes to `data/lake/lighter-testnet/`. Testnet has 4 perps with a few days of test activity, so use it for checking the testnet setup, not for training. The script paces requests (0.5 s by default); bursts of parallel requests trigger Lighter's human-verification page, so run one backfill at a time.

Collector/API inventory recorded 2026-10-16: candles down to 1m, starting 2025-06-04 for ETH, BTC, SOL, HYPE, and NEAR; mark-price candles; 1h funding; and a paginated public trade tape (`/api/v1/trades`, 100 trades per page, newest first). Order-book history is not available; it has to be recorded from the websocket. The general training loader does not read the lake yet, so existing runs and their dataset identity are unchanged. See [the exact coverage audit](context/lighter_data_lake.md), including the 31 funding gaps and later mark-price start.

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

This is the legacy top-level runner. It uses the Lighter-only feature stack and GLFT backtesting, and its model-prediction integration still needs verification. The primary v2 research path is:

```bash
uv run python -m src.experiments.staged_trial --config configs/config.yml
```

The v2 path writes multi-horizon predictions, meta-label candidates, directional alpha backtests, and per-trial artifacts. Its research validity work remains tracked in `TASKS.md`.

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

The older staged GLFT configuration remains an exploratory path. It makes Stage 1 model/hyperparameter artifacts and Stage 2 backtests from selected Stage 1 predictions. The starter smoke config runs one model trial and a small GLFT strategy grid; expand `staged_trial.stage1.search_space` or `staged_trial.stage2.strategy_search` for longer searches.

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

Trackio logging is enabled in `configs/config.yml` for the active v2 research path and writes to the local `ethpredict` project by default. Non-smoke runs now write a local delivery receipt and verify that their finished run can be read back. Check the local setup without training or trading:

```bash
uv run python -m scripts.trackio_doctor
```

The command prints a unique run name and receipt path. Inspect that run with `uv run trackio get run --project ethpredict --run <printed-run-name> --json`. The v2 trial records include checkpoint identity and path, out-of-fold forecast metrics by fold, separate simulated validation/test backtest metrics, and trial status. Trackio mirrors local run artifacts; it is not the source of truth for model promotion.

Launch the local dashboard with the Python entry point when CLI networking is awkward in sandboxed shells:

```bash
uv run python -c "import trackio; trackio.show(project='ethpredict', open_browser=False, host='127.0.0.1', server_port=7860)"
```

The CLI alternative is also available:

```bash
uv run trackio show --project ethpredict
```

### Archived experiment EDA

The separate `ethpredict_eda` project summarizes recorded runs without training. It reads surviving manifests and parquet files first, then uses Trackio metrics for missing local artifacts. Its custom frontend includes bundled Plotly JavaScript and does not need a CDN.

```bash
.venv/bin/python scripts/experiment_eda.py clean          # inspect exact IDs and local paths
.venv/bin/python scripts/experiment_eda.py clean --apply  # remove metadata-marked smoke/setup records
.venv/bin/python scripts/experiment_eda.py build          # rebuild distilled project and report
.venv/bin/python scripts/experiment_eda.py serve          # http://127.0.0.1:7861
```

Run the standard `ethpredict` dashboard on port 7860 to use the ledger's source-run links. The build writes `eda_frontend/data.json`, `artifacts/eda_report.md`, and `artifacts/eda_report.json`. The cleanup audit is `artifacts/eda_cleanup_inventory.json`. Rebuilds replace only runs in `ethpredict_eda`; the source experiment files and research runs are read only.

Read the dashboard in order: provenance, forecast quality, candidate/trade rate, costs and net PnL, validation stability, then archived test evidence. The ledger flags missing files, metric conflicts, zero trades, and possible final-test reuse. Legacy staged trials have their own cohort. The archive does not nominate a deployed model.


## Research roadmap

The active v2 pipeline already builds multi-horizon forecasts, candidate signals,
triple-barrier labels, held-out meta probabilities and directional alpha backtests.
GLFT remains a separate exploratory/passive-execution path. The earlier proposal
to build these layers is superseded by the merged label/split/training controls.

With PR #51 and its included #53 changes merged, recover funding gaps (#54), then add an explicit
lake reader or immutable export with partition provenance and coverage gates.
Specify one new mechanism/horizon before fitting: the tested continuation recipe
failed its economic gate. Keep fixed notional and a small recorded budget; require
positive after-cost development evidence, adequate trade/block coverage, cost
stress survival and feature credibility before a fresh predeclared final holdout.

#19 sizing, #22 benchmark trading parity and #43 → #44 → #45 current-state work
remain open. The 12 controlled experiment issues #30–#41 remain separate protocols.
See [the implementation plan](ethpredict-implementation-plan.md) and
[the AFML strategy reading/research plan](strategy-research-plan.md).

## Configuration

Use `configs/config.yml` and `configs/schema.yaml` as the executable v2 research
contract; validate through `src/config/loader.py`. Do not substitute the older
hierarchical/XGBoost/GLFT planning example for the current configuration.
`configs/lighter_experiments.yml` configures the separate comparison runner;
`configs/staged_trial_smoke.yml` retains the legacy staged smoke path.

Resolve and freeze config, code, raw inputs, split, seeds, costs and trial budget
for every study. Lake availability does not change the default input path or
enable side-data groups automatically. PR-specific commands/configs belong to
their implementation branch until merged.

## Verification

Runtime checks (not all were executed by this documentation update):

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

- The older configured quarterly raw Lighter export has 26,496 5m rows from `2026-03-28 00:00:00 UTC` through `2026-06-27 23:55:00 UTC`; 24-hour model lookbacks use 288 bars and the next-hour target uses 12 bars.
- Optional Lighter `exchangeMetrics` outputs are best-effort daily/all-period side data. Mark-price 5m, funding 1h, order-book snapshots, recent-trade snapshots, and exchange metrics stay under `data/lighter/`. They enter training features only through an enabled `data.side_data.groups` entry that passes its coverage gate; none is enabled by default.
- `requirements.txt` is preserved for legacy setup until a later cleanup; uv metadata is the active experiment dependency source.
- `archive/legacy_data_sources/README.md` lists archived files and why they are no longer active.
