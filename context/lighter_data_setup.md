# Lighter Data Setup for ETHPredict

This is the data-first Lighter integration. It collects public market data and
writes artifacts that the active Lighter-only training code can consume without
account keys or transaction signing.

## Current Scope

Implemented data endpoints:

- `/api/v1/orderBookDetails` to resolve and snapshot market metadata.
- `/api/v1/candles` for OHLCV training data. Lighter returns at most 500
  candles per call, so the collector chunks longer ranges.
- `/api/v1/markPriceCandles` for mark-price side data.
- `/api/v1/fundings` and `/api/v1/funding-rates` for funding features.
- `/api/v1/orderBookOrders` for a latest order book depth snapshot.
- `/api/v1/recentTrades` for a latest trades snapshot.
- `/api/v1/exchangeMetrics` for optional daily volume and open-interest side tables when Lighter accepts the requested period/filter combination.

ETH perp is configured as `market_id: 0`. The collector writes model-compatible
OHLCV rows to `data/raw/ETHUSDT-<resolution>-lighter-<start>-<end>.csv`.
`DataPreprocessor.load_price_data()` intentionally reads only files matching that
Lighter filename pattern.

## Install

The public SDK documented by Lighter currently lives at:

```bash
pip install git+https://github.com/elliottech/lighter-python.git
```

The data collector uses public REST endpoints directly because no account auth
or signer is needed for training data.

## Collect Training Data

Use the defaults from `configs/config.yml`:

```bash
python scripts/lighter_collect_data.py --config configs/config.yml
```

Or collect an explicit range:

```bash
python scripts/lighter_collect_data.py \
  --symbol ETH \
  --market-type perp \
  --market-id 0 \
  --resolution 1h \
  --start 2025-01-17 \
  --end 2025-02-01
```

Outputs:

- `data/raw/ETHUSDT-1h-lighter-20250117-20250201.csv`
- `data/lighter/candles_ETH_perp_1h_20250117_20250201.csv`
- `data/lighter/mark_price_candles_ETH_perp_1h_20250117_20250201.csv`
- `data/lighter/fundings_ETH_perp_1h_20250117_20250201.csv`
- `data/lighter/funding_rates_latest.csv`
- `data/lighter/order_book_details_ETH_perp.json`
- `data/lighter/order_book_orders_ETH_perp_latest.csv`
- `data/lighter/recent_trades_ETH_perp_latest.csv`
- Optional: `data/lighter/exchange_metric_volume_ETH_perp_daily.csv`
- Optional: `data/lighter/exchange_metric_open_interest_ETH_perp_daily.csv`

## Training Pipeline

After collection, the active feature stack can build OHLCV-only sequence features from the raw Lighter CSV:

```bash
python -m src.data.features_all full_features --data-dir data --sequence-length 24
```

The side-data CSVs are intentionally saved separately. Funding, mark price,
open interest, order-book depth, and recent trades are not joined into active
features in this pass.

## Notes

- Lighter public REST rate limits are strict for standard accounts. The
  collector sleeps between paged candle requests via `request_pause_seconds`.
- Lighter mainnet ETH market history starts around January 17, 2025. Avoid
  requesting pre-genesis training ranges from Lighter.
- Trading, account state, API key creation, nonce management, and signed
  `sendTx` calls are deliberately out of this first data pass.
