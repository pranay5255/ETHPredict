"""
binance_data.py
Download and process historical Ethereum market data from Binance's public data repository
for May and June 2025.

Outputs: Processed CSVs in ./data
Dependencies: pandas, requests, zipfile, io
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
import zipfile
import io

import pandas as pd
import requests

# ---------- setup ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Base URL for Binance's public data
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1h"

def download_and_extract_zip(url: str, target_dir: Path) -> None:
    """Download and extract a zip file from the given URL."""
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"✓ Extracted {url}")
    except Exception as e:
        print(f"Error processing {url}: {e}", file=sys.stderr)

def process_historical_data():
    """Download and process historical data for May and June 2025."""
    # Create directories for raw and processed data
    raw_dir = DATA_DIR / "raw"
    processed_dir = DATA_DIR / "processed"
    raw_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

    # Files to download (May and June 2025)
    files_to_download = [
        "ETHUSDT-1h-2025-05.zip",
        "ETHUSDT-1h-2025-04.zip"
    ]

    # Download and extract files
    for filename in files_to_download:
        url = f"{BASE_URL}/{filename}"
        download_and_extract_zip(url, raw_dir)

    # Process the extracted CSV files
    for csv_file in raw_dir.glob("*.csv"):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file, header=None)
            
            # Rename columns according to Binance's format
            df.columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_base_volume",
                "taker_quote_volume", "ignore"
            ]
            
            # Convert timestamps to datetime
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            
            # Save processed file
            output_file = processed_dir / f"processed_{csv_file.name}"
            df.to_csv(output_file, index=False)
            print(f"✓ Processed {csv_file.name} -> {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}", file=sys.stderr)

if __name__ == "__main__":
    process_historical_data() 