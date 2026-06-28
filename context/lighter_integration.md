# Lighter Integration Guide for ETHPredict

The active repository scope is public Lighter market data only. Legacy Binance,
DeFiLlama, Santiment, DEX simulation, and bribe-optimization planning notes are
archived under `archive/legacy_data_sources`.

## Active Data Path

1. Configure `configs/config.yml` with `data.sources: [lighter]` and the
   `data.lighter` market/date settings.
2. Collect data with:

```bash
python scripts/lighter_collect_data.py --config configs/config.yml
```

3. Keep the model OHLCV export in `data/raw/ETHUSDT-<resolution>-lighter-<start>-<end>.csv`.
4. Build OHLCV-only features with:

```bash
python -m src.data.features_all full_features --data-dir data --sequence-length 24
```

`DataPreprocessor` ignores non-Lighter raw CSV names. Lighter side-data files in
`data/lighter/` are preserved for later feature work, but the active model path
uses candles only.

## Signed Testnet Trading Setup

The repo now has a separate signed-trading setup path that keeps public data
collection isolated from authenticated order flow.

- Non-secret defaults live in `configs/lighter_trading.yml`.
- Local secrets and account identifiers live in `.env`.
- `.env.example` documents the required variables.
- `scripts/lighter_trading_setup.py env` lists the required and optional
  environment variables.
- `scripts/lighter_trading_setup.py check` verifies the testnet endpoint,
  account, API key, nonce, auth token, and active-orders read without placing
  an order.
- `scripts/lighter_trading_setup.py register-api-key` can create and register a
  Lighter API key on testnet when `LIGHTER_ETH_PRIVATE_KEY` is present.
- `scripts/lighter_place_test_order.py` prepares a guarded testnet limit order
  and only submits with `--submit`.

Required for normal trading:

- `LIGHTER_ACCOUNT_INDEX`
- `LIGHTER_API_KEY_INDEX`
- `LIGHTER_API_PRIVATE_KEY`

Required only for API key setup or rotation:

- `LIGHTER_ETH_PRIVATE_KEY`

The default REST endpoint is `https://testnet.zklighter.elliot.ai`; websocket
subscriptions use `wss://testnet.zklighter.elliot.ai/stream`. The SDK uses
chain ID `300` for non-mainnet URLs.

Subaccount control should use two env roles:

- Admin/source env for one-time subaccount creation and USDC transfers. This
  can use a source account API key, but should not be used by experiments.
- Bot env for the funded subaccount. This contains only the subaccount index and
  subaccount API key and is the only env used for programmatic trading.

Do not keep the L1 wallet private key in repo env files. Use it only for
short-lived API-key registration/rotation, then remove it. Mainnet writes require
separate mainnet env files plus explicit mainnet confirmation flags.

## Still Out of Scope

- Binance, DeFiLlama, and Santiment joins.
- DEX simulation configuration and code paths.
- Bribe or MEV optimization configuration.
- Production/mainnet trading automation.
