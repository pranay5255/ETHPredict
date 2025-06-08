# ETH Price Prediction Pipeline

This repository contains a complete pipeline for Ethereum price prediction, including data collection, preprocessing, and model training.

## Data Sources

The pipeline collects data from seven different sources:
2. Binance (spot candles & trades)
3. Santiment (social & on-chain metrics)

6. Binance Bulk Data (funding & open interest)
7. DeFiLlama (TVL & protocol data)

## Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd ETHPredict
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   GLASSNODE_API_KEY=your_glassnode_api_key
   SANAPI_TOKEN=your_santiment_api_token
   ```

## Usage

### 1. Data Collection

#### Main Data Collection
Run the main data collection script:
```bash
python get_eth_data.py
```

#### DeFiLlama Data Collection
For DeFiLlama specific data (April-May 2025):
```bash
python defillama_data.py
```

The scripts will:
- Create a `data` directory if it doesn't exist
- Fetch data from each source
- Save the data as CSV/Parquet files in the `data` directory
- Handle API rate limits and errors gracefully

### 2. Data Preprocessing

Run the preprocessing pipeline:
```bash
python setup.py
```

This will:
- Validate the data structure
- Preprocess the collected data
- Create feature sets for model training
- Handle timezone consistency
- Generate training datasets

### 3. Model Training

Train the ensemble model:
```bash
python train.py
```

The training process includes:
- Hierarchical model architecture (Level-0, Level-1, Level-2)
- Triple-barrier labeling with meta-labeling
- Purged cross-validation with embargo
- Sample uniqueness weighting
- Risk-adjusted performance metrics

## Output Files

### Data Collection Outputs
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

### Model Outputs
The training process generates:
- Trained model weights in the `models` directory
- Performance metrics and validation results
- Cross-validation scores
- Feature importance analysis

## Notes

- Some data sources require API keys (Glassnode, Santiment)
- Others are public and don't require authentication
- The scripts include error handling for API failures
- Data is collected at hourly granularity where available, falling back to daily where necessary
- DeFiLlama data is collected with a 1 request/second rate limit
- The preprocessing pipeline handles timezone consistency automatically
- Model training includes comprehensive validation and performance metrics

## Troubleshooting

If you encounter timezone-related errors during preprocessing:
1. Ensure all data files are in the correct format
2. Check that timestamps are consistent across datasets
3. Run the preprocessing pipeline with verbose logging:
   ```bash
   python setup.py --verbose
   ```

For model training issues:
1. Verify that the preprocessed data is available
2. Check GPU availability if using CUDA
3. Monitor memory usage during training
4. Review the training logs for specific errors 