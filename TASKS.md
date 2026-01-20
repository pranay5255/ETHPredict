# ETHPredict Execution + Backtesting Task Phases

## Phase 0 - Preconditions and Critical Fixes
- [ ] Resolve `csv_loader` import (create `src/csv_loader.py` or fix `runner.py` import to `src.main.validate_csvs`)
- [ ] Verify `configs/schema.yaml` exists and matches `configs/config.yml`
- [ ] Replace backtest placeholder predictions with real model inference in `runner.py`
- [ ] Confirm DEX simulation is out of scope for now (price-data backtest only)

## Phase 1 - Dev Environment (Docker)
- [ ] Start dev environment with `docker-compose-dev.yml` (note: no proxy server)
- [ ] Validate dev frontend + backend are reachable after compose up
- [ ] Record service URLs/ports and log locations for debugging

## Phase 2 - Data Readiness and Validation
- [ ] Place Binance OHLCV CSVs in `data/raw/`
- [ ] Place DeFiLlama TVL CSVs in `data/`
- [ ] Place Santiment CSVs in `data/`
- [ ] Run CSV validation (via `runner.py` or `src/main.py`) and confirm rejects go to reject dir

## Phase 3 - Training and Inference Prep
- [ ] Run training pipeline (`src/training/trainer.py`) with target config
- [ ] Ensure model artifacts are saved and loadable for inference
- [ ] Capture training metrics (loss curves, key performance stats)

## Phase 4 - Backtesting Execution
- [ ] Run pipeline end-to-end (`python runner.py configs/config.yml`)
- [ ] Ensure backtest uses model predictions, not shifted prices
- [ ] Compute metrics (Sharpe, drawdown, IR, directional accuracy) and save outputs
- [ ] Validate outputs in `results/` and `logs/`

## Phase 5 - Live Execution (Lighter Integration)
- [ ] Create API keys and set `API_KEY_PRIVATE_KEY`, `BASE_URL`, `ACCOUNT_INDEX`
- [ ] Initialize `SignerClient` and confirm auth token creation
- [ ] Implement nonce management via `TransactionApi.next_nonce`
- [ ] Build signal-to-order mapper (prediction + confidence -> order type/size/TIF)
- [ ] Implement execution client: sign orders and send `send_tx` / `send_tx_batch`
- [ ] Add risk gates (max exposure, drawdown checks, confidence thresholds)
- [ ] Add websocket listeners for orderbook/account updates
- [ ] Test end-to-end on Lighter testnet before mainnet

## Phase 6 - Reporting and Validation
- [ ] Review results summary in `results/` (CSV/JSON outputs)
- [ ] Verify model and backtest performance meets thresholds in `PROJECT_STATUS.md`
- [ ] Document any gaps before production deployment
