from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_bars(df: pd.DataFrame, bar_type: str) -> pd.DataFrame:
    if bar_type == "tick":
        threshold = 1000
        metric = pd.Series(range(len(df))) + 1
    elif bar_type == "volume":
        threshold = 10000
        metric = df["volume"].cumsum()
    else:  # dollar
        threshold = 10_000_000
        metric = (df["close"] * df["volume"]).cumsum()

    groups = metric // threshold
    bars = df.groupby(groups).agg(
        timestamp=("timestamp", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return bars.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--bar-type", default="tick")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(
        args.csv,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume"],
        usecols=range(6),
    )
    bars = sample_bars(df, args.bar_type)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(args.out)


if __name__ == "__main__":
    main()
