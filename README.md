# ETH Price Prediction Data Collection

This repository contains scripts to collect Ethereum-related data from various sources for price prediction analysis.

## Data Sources

The scripts collect data from seven different sources:
1. Glassnode (on-chain & market metrics)
2. Binance (spot candles & trades)
3. Santiment (social & on-chain metrics)
4. CoinGecko (market chart)
5. Funding Rate (perpetual funding rates)
6. Binance Bulk Data (funding & open interest)
7. DeFiLlama (TVL & protocol data)

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

### Main Data Collection
Run the main data collection script:
```bash
python get_eth_data.py
```

### DeFiLlama Data Collection
For DeFiLlama specific data (April-May 2025):
```bash
python defillama_data.py
```

The scripts will:
- Create a `data` directory if it doesn't exist
- Fetch data from each source
- Save the data as CSV/Parquet files in the `data` directory
- Handle API rate limits and errors gracefully

## Output Files

### Main Data Collection Outputs
The main script generates the following CSV files in the `data` directory:
- `glassnode_price_1h.csv`
- `glassnode_volume_1h.csv`
- `binance_ethusdt_klines_1h.csv`
- `santiment_social_1h.csv`
- `santiment_dev_1h.csv`
- `coingecko_price_1h.csv`
- `fundingrate_binance_eth_1h.csv`

Additionally, it downloads ZIP files for Binance bulk data:
- `premiumIndex.zip` (funding rates)
- `openInterest.zip` (open interest)

### DeFiLlama Data Collection Outputs
The DeFiLlama script generates:
- `defillama_eth_chain_tvl_2025_04-05.csv` (Ethereum chain TVL)
- `defillama_eth_top10_snapshots_2025_04-05.csv` (Top-10 protocols snapshot)
- `defillama_eth_top10_tvl_2025_04-05.parquet` (Daily TVL history for top protocols)

## Notes

- Some data sources require API keys (Glassnode, Santiment)
- Others are public and don't require authentication
- The scripts include error handling for API failures
- Data is collected at hourly granularity where available, falling back to daily where necessary
- DeFiLlama data is collected with a 1 request/second rate limit 