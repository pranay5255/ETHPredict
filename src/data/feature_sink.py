from pathlib import Path
import pandas as pd


def save_features(asset: str, bar_type: str, df: pd.DataFrame) -> Path:
    date = pd.to_datetime(df["timestamp"].iloc[0], unit="ms").strftime("%Y%m%d")
    out_dir = Path("build/features") / asset / bar_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.parquet"
    df.to_parquet(out_path)
    return out_path
