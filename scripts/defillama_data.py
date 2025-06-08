# defillama_data.py
"""
DeFiLlama grabber – **April & May 2025** (API‑spec compliant)
============================================================
• Uses only endpoints present in the official OpenAPI spec shipped with your
  `defillama-api-spec.json` upload (v1.0.0‑oas3).
• Window fixed to **1 Apr 2025 → 31 May 2025**.
• Saves:
  ‑ `defillama_eth_chain_tvl_2025_04‑05.csv`  (Ethereum chain TVL)
  ‑ `defillama_eth_top10_snapshots_2025_04‑05.csv`  (Top‑10 protocols, 31 May)
  ‑ `defillama_eth_top10_tvl_2025_04‑05.parquet`  (daily TVL history)

Run:
```bash
pip install pandas requests tqdm python-dotenv
python defillama_data.py
```
No key required. 1 req/sec polite delay.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Fixed date window (UTC)
# ──────────────────────────────────────────────────────────────────────────────
START = datetime(2025, 4, 1, tzinfo=timezone.utc)
END   = datetime(2025, 5, 31, 23, 59, 59, tzinfo=timezone.utc)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
BASE = "https://api.llama.fi"
HEADERS = {"User-Agent": "defillama-data-script/1.2"}

# polite
SLEEP = 1.0  # seconds between calls (limits: ~20 r/s free, we use 1 r/s)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def dump(df: pd.DataFrame, name: str):
    fp = DATA_DIR / f"{name}.csv"
    df.to_csv(fp, index=False)
    print(f"✔ {fp.name}  ({len(df):,} rows)")


def _get(path: str):
    """Light wrapper around GET."""
    url = f"{BASE}/{path.lstrip('/')}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(SLEEP)
    return resp.json()

# ──────────────────────────────────────────────────────────────────────────────
# Endpoint wrappers – all come from the OpenAPI spec
# ──────────────────────────────────────────────────────────────────────────────

def list_protocols() -> pd.DataFrame:
    return pd.DataFrame(_get("protocols"))


def historical_chain_tvl(chain: str = "Ethereum") -> pd.DataFrame:
    data = _get(f"v2/historicalChainTvl/{chain}")
    df = pd.DataFrame(data)
    df.rename(columns={"date": "timestamp", "tvl": "tvl_usd"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.loc[(df["timestamp"] >= START) & (df["timestamp"] <= END)]


def protocol_overview(slug: str) -> Dict:
    """/protocol/{slug} – includes full TVL series & meta."""
    return _get(f"protocol/{slug}")


def historical_protocol_tvl(slug: str) -> pd.DataFrame:
    js = protocol_overview(slug)
    if "tvl" not in js:
        raise ValueError("Unexpected payload – 'tvl' key missing")
    df = pd.DataFrame(js["tvl"], columns=["timestamp", "tvl_usd"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.loc[(df["timestamp"] >= START) & (df["timestamp"] <= END)]

# ──────────────────────────────────────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────────────────────────────────────

def filter_ethereum_protocols(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["chains"].apply(lambda chains: "Ethereum" in chains if isinstance(chains, list) else False)
    return df[mask]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1️⃣ Ethereum chain TVL for April‑May 2025
    eth_chain = historical_chain_tvl()
    dump(eth_chain, "defillama_eth_chain_tvl_2025_04-05")

    # 2️⃣ Discover top‑10 ETH protocols by TVL as of 31 May 2025
    all_proto = list_protocols()
    eth_proto = filter_ethereum_protocols(all_proto)
    snapshot_date = END  # 31 May 2025
    eth_proto_sorted = eth_proto.sort_values("tvl", ascending=False).head(10).reset_index(drop=True)

    # 3️⃣ Save snapshot meta
    snap_cols = [
        "name", "slug", "symbol", "category", "tvl",
        "mcap", "fdv", "change_1d", "change_7d", "change_1m", "chains",
    ]
    snapshot = eth_proto_sorted.reindex(columns=snap_cols)
    snapshot["snapshot_date"] = snapshot_date.isoformat()
    dump(snapshot, "defillama_eth_top10_snapshots_2025_04-05")

    # 4️⃣ Fetch per‑protocol TVL series within window
    tvl_frames: List[pd.DataFrame] = []
    for slug in tqdm(snapshot["slug"], desc="Protocol TVLs"):
        try:
            df = historical_protocol_tvl(slug)
            df["protocol"] = slug
            tvl_frames.append(df)
        except Exception as e:
            print(f"⚠️  {slug}: {e}", file=sys.stderr)
    if tvl_frames:
        hist = pd.concat(tvl_frames, ignore_index=True)
        pq_path = DATA_DIR / "defillama_eth_top10_tvl_2025_04-05.parquet"
        hist.to_parquet(pq_path, index=False)
        print(f"✔ {pq_path.name}  ({len(hist):,} rows)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
