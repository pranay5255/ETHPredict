# Lighter Integration Guide for ETHPredict

Update: public Lighter data collection is now implemented. See `context/lighter_data_setup.md` for the current data-first collector, commands, and output files. The sections below remain as execution/backtesting planning notes.

This doc answers common integration questions using current repo capabilities and
the Lighter "Get Started for Programmers" guide.

## How do I get data to fetch for the models in this repo?

Current pipeline expects CSV inputs and uses the feature engineering stack in
`src/data/features_all.py` and `DataPreprocessor` (see `PROJECT_STATUS.md`).

- **Price data**: Binance OHLCV CSVs in `data/raw/` (example files exist).
- **On-chain TVL**: DeFiLlama CSVs in `data/`.
- **Social/network metrics**: Santiment CSVs in `data/`.
- **Validation**: CSV validation is performed by `runner.py`/`src/main.py`.

Minimum steps (offline):
1. Place your CSVs into `data/raw/` and `data/` with the same schema as the
   existing samples.
2. Run the pipeline: `python runner.py configs/config.yml`.

Real-time data collection is **not implemented** yet (see `PROJECT_STATUS.md`).
If you want live data, add API clients (Binance/DeFiLlama/Santiment) and write
CSV or direct ingestion into the `DataPreprocessor` pipeline.

## How do I integrate Lighter APIs to execute orders for my model predictions?

From the Lighter "Get Started for Programmers" guide:

1. **Create API keys**:
   - Generate `API_KEY_PRIVATE_KEY`.
   - Use the correct `BASE_URL` for testnet or mainnet.
   - Fetch `ACCOUNT_INDEX` via `AccountApi.account`.
2. **Initialize SignerClient**:
   - `SignerClient` signs transactions (`create/modify/cancel` orders).
3. **Handle nonce**:
   - Use `TransactionApi.next_nonce` or manage nonce increments per API key.
4. **Sign + send transactions**:
   - Sign with `SignerClient.sign_create_order` (or helpers like
     `create_market_order`, `create_cancel_order`).
   - Send via `TransactionApi.send_tx` or `send_tx_batch`.
5. **Market data**:
   - Use `OrderApi.order_book_details` or `OrderApi.order_books`.
   - For streaming updates, connect via WebSockets and use auth token from
     `SignerClient.create_auth_token_with_expiry`.

Integration pattern for this repo:
- **Model inference service**: run predictions on feature windows.
- **Signal-to-order mapper**: translate prediction + confidence to order type,
  size, and time-in-force (e.g., post-only limit orders).
- **Execution client**: encapsulate Lighter API setup, nonce management, and
  `send_tx` operations.

## What is the optimal strategy to set this up?

For this repo and Lighter's API workflow, the most robust setup is:

- **Offline training + backtesting first** using the existing pipeline, then
  transition to live execution after sanity checks.
- **Start on testnet** with small sizes and monitor nonce/order lifecycle.
- **Use limit/post-only orders** for maker-style execution; reserve market/taker
  orders for stop-loss or emergency exits.
- **Risk gating**: enforce max exposure, drawdown checks, and confidence
  thresholds before sending orders.
- **Split responsibilities**:
  - `Data/Training` in this repo.
  - `Execution` as a small service that reads predictions and sends orders.
- **Use websockets** for realtime orderbook/account updates and local state.

## What are the steps to backtest a strategy and then deploy the same strategy on Lighter?

**Backtest (offline)**
1. Prepare CSV inputs in `data/raw/` and `data/`.
2. Run the pipeline: `python runner.py configs/config.yml`.
3. Ensure backtesting uses **model predictions**, not placeholder shifted prices
   (identified as a partial integration in `PROJECT_STATUS.md`).
4. Evaluate metrics, risk profile, and stability.
5. Export the final strategy config (thresholds, position sizing, order rules).

**Deploy (Lighter testnet → mainnet)**
1. Create API keys and find `ACCOUNT_INDEX` using `AccountApi`.
2. Set `BASE_URL` for testnet; initialize `SignerClient`.
3. Build an execution runner that:
   - Loads model outputs.
   - Maps prediction → order parameters.
   - Uses `next_nonce` + `send_tx` to place/cancel orders.
4. Stream orderbook/account updates via websockets and maintain local state.
5. Verify end-to-end consistency (fills, PnL, inventory tracking).
6. Move to mainnet by switching `BASE_URL` and using mainnet API keys.

## Practical Gaps to Address in ETHPredict

Based on `PROJECT_STATUS.md`:
- **Backtesting integration** still uses placeholder predictions.
- **Real-time data collection** is missing.
- **Production deployment tooling** is incomplete.

If you want, I can implement:
- A minimal Lighter execution client module.
- A signal-to-order adapter tied to `HierarchicalPredictor`.
- A backtesting fix to use real model inference output.

