# ETH Price Prediction Data Collection

This repository contains a script to collect Ethereum-related data from various sources for price prediction analysis.

## Data Sources

The script collects data from seven different sources:
1. Glassnode (on-chain & market metrics)
2. Binance (spot candles & trades)
3. Santiment (social & on-chain metrics)
4. CoinGecko (market chart)
5. Funding Rate (perpetual funding rates)
6. Binance Bulk Data (funding & open interest)
7. DeFiLlama (TVL & flows)

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   GLASSNODE_API_KEY=your_glassnode_api_key
   SANAPI_TOKEN=your_santiment_api_token
   ```

## Usage

Run the data collection script:
```bash
python get_eth_data.py
```

The script will:
- Create a `data` directory if it doesn't exist
- Fetch one month of hourly (or daily) data from each source
- Save the data as CSV files in the `data` directory
- Handle API rate limits and errors gracefully

## Output Files

The script generates the following CSV files in the `data` directory:
- `glassnode_price_1h.csv`
- `glassnode_volume_1h.csv`
- `binance_ethusdt_klines_1h.csv`
- `santiment_social_1h.csv`
- `santiment_dev_1h.csv`
- `coingecko_price_1h.csv`
- `fundingrate_binance_eth_1h.csv`
- `defillama_tvl_daily.csv`

Additionally, it downloads ZIP files for Binance bulk data:
- `premiumIndex.zip` (funding rates)
- `openInterest.zip` (open interest)

## Notes

- Some data sources require API keys (Glassnode, Santiment)
- Others are public and don't require authentication
- The script includes error handling for API failures
- Data is collected at hourly granularity where available, falling back to daily where necessary 