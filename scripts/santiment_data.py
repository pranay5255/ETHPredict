"""
santiment_data.py
Fetch Ethereum on-chain and social metrics from Santiment API.

Outputs: CSVs in ./data
Dependencies: pandas, san, python-dotenv
"""

import os, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import san
from dotenv import load_dotenv

# ---------- setup ----------
load_dotenv()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Set date range for April-May 2025
START_DATE = "2025-04-01"
END_DATE = "2025-05-31"

def dump(df: pd.DataFrame, name: str):
    fp = DATA_DIR / f"{name}.csv"
    df.to_csv(fp, index=False)
    print(f"✔ Saved {fp} ({len(df):,} rows)")

def fetch_santiment_metrics():
    # Set API key
    san.ApiConfig.api_key = os.getenv("SANAPI_TOKEN")
    
    # Define metrics to fetch - selected key metrics for Ethereum
    metrics = [
        "price_usd",               # Price data
        "volume_usd",              # Trading volume
        "marketcap_usd",           # Market capitalization
        "daily_active_addresses",  # On-chain activity
        "dev_activity",            # Development activity
        "social_volume_total",     # Social media mentions
        "social_dominance_total",  # Social dominance
        "transaction_volume",      # On-chain transaction volume
        "network_growth",          # New addresses
        "velocity",                # Token velocity
        "mvrv_usd",               # Market Value to Realized Value
        "mean_age",               # Mean coin age
        "mean_dollar_invested_age", # Mean dollar invested age
        "realized_value_usd",     # Realized value
        "total_supply",           # Total supply
        "supply_on_exchanges",    # Supply on exchanges
        "supply_outside_exchanges" # Supply outside exchanges
    ]
    
    # Fetch data for each metric
    dfs = {}
    for metric in metrics:
        try:
            df = san.get(
                metric,
                slug="ethereum",
                from_date=START_DATE,
                to_date=END_DATE,
                interval="1d"
            )
            # Rename the value column to include the metric name
            df.columns = [f"{metric}_value"]
            dfs[metric] = df
            print(f"✔ Fetched {metric}")
        except Exception as e:
            print(f"Error fetching {metric}:", e, file=sys.stderr)
    
    # Merge all metrics into one DataFrame
    if dfs:
        # Start with the first metric
        final_df = dfs[metrics[0]]
        # Merge remaining metrics
        for metric in metrics[1:]:
            if metric in dfs:
                final_df = final_df.merge(
                    dfs[metric],
                    left_index=True,
                    right_index=True,
                    how="outer"
                )
        
        # Reset index to make date a column
        final_df.reset_index(inplace=True)
        final_df.rename(columns={"index": "date"}, inplace=True)
        
        return final_df
    return None

if __name__ == "__main__":
    try:
        santiment_data = fetch_santiment_metrics()
        if santiment_data is not None:
            dump(santiment_data, "santiment_metrics_april_may_2025")
    except Exception as e:
        print("Santiment error:", e, file=sys.stderr) 