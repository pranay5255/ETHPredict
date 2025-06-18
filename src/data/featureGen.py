import pandas as pd


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    df["return"] = df["close"].pct_change().fillna(0)
    df["rolling_vol"] = df["return"].rolling(24).std().fillna(0)
    return df
