"""
get_eth_data.py
Main script to fetch Ethereum-related data from multiple sources.
This script orchestrates the data collection from:
1. Binance (market data)
2. Santiment (on-chain & social metrics)
3. DeFiLlama (TVL & protocols)

Outputs: CSVs in ./data
Dependencies: See individual module requirements
"""

import os, sys
from pathlib import Path

# Import data collection modules
from binance_data import fetch_binance_market_data
from santiment_data import fetch_santiment_metrics
from defillama_data import fetch_defillama

# ---------- setup ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def main():
    print("Starting Ethereum data collection...")
    
    # 1. Fetch Binance market data
    print("\nFetching Binance market data...")
    fetch_binance_market_data()
    
    # 2. Fetch Santiment metrics
    print("\nFetching Santiment metrics...")
    santiment_data = fetch_santiment_metrics()
    if santiment_data is not None:
        santiment_data.to_csv(DATA_DIR / "santiment_metrics_daily.csv", index=False)
    
    # 3. Fetch DeFiLlama data
    print("\nFetching DeFiLlama data...")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    try:
        price_now, tvl_df, eth_protocols = fetch_defillama()
        print(f"DeFiLlama current price snapshot: {price_now:,.2f} USD")
        tvl_df.to_csv(DATA_DIR / "defillama_tvl_daily.csv", index=False)
        eth_protocols.to_csv(DATA_DIR / "defillama_eth_protocols.csv", index=False)
    except Exception as e:
        print("DeFiLlama error:", e, file=sys.stderr)
    
    print("\nDone. Join the CSVs on timestamp (or date) to create your feature set 🧩")

if __name__ == "__main__":
    main()                                                                                                                                                                                                                                                                                          