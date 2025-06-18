import argparse
import os
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# --- Bar Sampling Logic ---
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

# --- Simple Feature Generation ---
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    df["return"] = df["close"].pct_change().fillna(0)
    df["rolling_vol"] = df["return"].rolling(24).std().fillna(0)
    return df

# --- Feature Saving Utility ---
def save_features(asset: str, bar_type: str, df: pd.DataFrame) -> Path:
    date = pd.to_datetime(df["timestamp"].iloc[0], unit="ms").strftime("%Y%m%d")
    out_dir = Path("build/features") / asset / bar_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.parquet"
    df.to_parquet(out_path)
    return out_path

# --- Full Feature Engineering and Preprocessing ---
class DataPreprocessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.granularities = ["30s", "1m", "5m", "1h"]

    def load_price_data(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for gran in self.granularities:
            candidates = list(self.data_dir.glob(f"ETHUSDT-{gran}-*.csv")) + \
                         list((self.data_dir / "raw").glob(f"ETHUSDT-{gran}-*.csv"))
            frames = []
            binance_columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            for csv_path in candidates:
                try:
                    df = pd.read_csv(csv_path, header=None, names=binance_columns)
                    df["timestamp"] = pd.to_datetime(df["open_time"], unit='ms', errors='coerce')
                    if df["timestamp"].dt.tz is not None:
                        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
                    frames.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {csv_path}: {e}")
            if frames:
                price = pd.concat(frames, ignore_index=True)
                price = price.drop_duplicates(subset="timestamp").sort_values("timestamp")
                data[gran] = price
        return data

    def load_chain_tvl(self) -> Optional[pd.DataFrame]:
        chain_tvl_files = list(self.data_dir.glob("defillama_eth_chain_tvl*.csv"))
        for f in chain_tvl_files:
            df = pd.read_csv(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            return df
        return None

    def load_santiment(self) -> Optional[pd.DataFrame]:
        santiment_files = list(self.data_dir.glob("santiment_metrics*.csv"))
        for f in santiment_files:
            df = pd.read_csv(f)
            if "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"])
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            return df
        return None

    def fracdiff(self, series: pd.Series, d: float, thres: float = 0.01) -> pd.Series:
        w = [1.0]
        for k in range(1, len(series)):
            w_ = -w[-1] * (d - k + 1) / k
            if abs(w_) < thres:
                break
            w.append(w_)
        w = np.array(w[::-1])
        output = series.copy() * np.nan
        for idx in range(len(w)-1, len(series)):
            output.iloc[idx] = np.dot(w, series.iloc[idx-len(w)+1:idx+1])
        return output

    def find_optimal_d(self, series: pd.Series, max_d: float = 1.0, corr_threshold: float = 0.97) -> float:
        best_d = 0.0
        for d in np.arange(0.1, max_d + 0.01, 0.1):
            diffed = self.fracdiff(series, d).dropna()
            if len(diffed) < 10:
                continue
            try:
                adf_stat, pvalue, *_ = adfuller(diffed)
                corr = np.corrcoef(series.dropna()[-len(diffed):], diffed)[0, 1]
                if pvalue < 0.05 and corr > corr_threshold:
                    best_d = d
                    break
            except Exception:
                continue
        return best_d

    def compute_entropy(self, returns: pd.Series, window: int = 24) -> pd.Series:
        def shannon_entropy(x):
            hist, _ = np.histogram(x, bins=10, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))
        return returns.rolling(window, min_periods=window).apply(shannon_entropy, raw=True)

    def cusum_flag(self, series: pd.Series, threshold: float = 2.0) -> pd.Series:
        mean_val = series.mean()
        cusum = (series - mean_val).cumsum()
        flags = (np.abs(cusum) > threshold * series.std()).astype(int)
        return flags

    def sadf_flag(self, series: pd.Series, min_window: int = 24, max_window: int = 100) -> pd.Series:
        flags = pd.Series(0, index=series.index)
        for i in range(max_window, len(series)):
            stats = []
            for w in range(min_window, max_window+1, 10):
                window_data = series.iloc[i-w+1:i+1]
                if window_data.isnull().any() or len(window_data) < w:
                    continue
                try:
                    adf_stat, *_ = adfuller(window_data)
                    stats.append(adf_stat)
                except Exception:
                    continue
            if stats and max(stats) > -1.0:
                flags.iloc[i] = 1
        return flags

    def volatility_regime(self, returns: pd.Series, window: int = 24) -> pd.Series:
        vol = returns.rolling(window).std()
        low, high = vol.quantile(0.33), vol.quantile(0.67)
        regime = pd.Series(1, index=vol.index)
        regime[vol <= low] = 0
        regime[vol >= high] = 2
        return regime

    def parkinson_vol(self, high: pd.Series, low: pd.Series, window: int = 24) -> pd.Series:
        pv = np.sqrt(0.25 * np.log(2) * (np.log(high / low) ** 2))
        return pv.rolling(window).mean()

    def prepare_features(self, granularity: str, sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        price_data = self.load_price_data().get(granularity)
        chain_tvl = self.load_chain_tvl()
        santiment = self.load_santiment()

        if price_data is None or chain_tvl is None:
            raise ValueError(f"Price and chain TVL data required for {granularity}")

        price_data = price_data.set_index("timestamp").sort_index()
        freq = pd.infer_freq(price_data.index[:100]) or granularity
        chain_hourly = chain_tvl.set_index("timestamp").resample(freq).ffill()
        if santiment is not None:
            santiment_hourly = santiment.set_index("timestamp").resample(freq).ffill()
        else:
            santiment_hourly = pd.DataFrame(index=chain_hourly.index)

        merged = (
            price_data
            .join(chain_hourly[["tvl_usd"]], how="left")
            .join(santiment_hourly, how="left")
            .sort_index()
        )
        merged.fillna(method="ffill", inplace=True)

        merged["price_return"] = np.log(merged["close"]).diff().fillna(0)
        merged["tvl_change"] = merged["tvl_usd"].pct_change().fillna(0)
        merged["price_tvl_ratio"] = merged["close"] / merged["tvl_usd"]
        merged["return_vol_24"] = merged["price_return"].rolling(24).std().fillna(0)
        merged["volume_tvl_ratio"] = merged["volume"] / merged["tvl_usd"]

        merged["address_growth"] = merged.get("daily_active_addresses_value", pd.Series(0, index=merged.index)).pct_change().fillna(0)
        merged["social_dominance_change"] = merged.get("social_dominance_total_value", pd.Series(0, index=merged.index)).pct_change().fillna(0)
        merged["mcap_tvl_ratio"] = merged.get("marketcap_usd_value", pd.Series(0, index=merged.index)) / merged["tvl_usd"]

        close_d = self.find_optimal_d(merged["close"])
        merged["fracdiff_close"] = self.fracdiff(merged["close"], close_d)
        merged["return_entropy_24"] = self.compute_entropy(merged["price_return"], 24)
        merged["return_entropy_168"] = self.compute_entropy(merged["price_return"], 168)
        merged["cusum_flag"] = self.cusum_flag(merged["price_return"])
        merged["sadf_flag"] = self.sadf_flag(merged["close"])
        merged["vol_regime"] = self.volatility_regime(merged["price_return"])
        merged["return_vol_1"] = merged["price_return"].rolling(1).std().fillna(0)
        merged["return_vol_168"] = merged["price_return"].rolling(168).std().fillna(0)
        merged["parkinson_vol"] = self.parkinson_vol(merged["high"], merged["low"], 24).fillna(0)

        feature_cols = [
            "close", "volume", "tvl_usd", "price_return", "tvl_change", "price_tvl_ratio",
            "return_vol_24", "volume_tvl_ratio", "address_growth", "social_dominance_change",
            "mcap_tvl_ratio", "fracdiff_close", "return_entropy_24", "return_entropy_168",
            "cusum_flag", "sadf_flag", "vol_regime", "return_vol_1", "return_vol_168", "parkinson_vol"
        ]
        features = merged[feature_cols].fillna(0).values
        targets = merged[["tvl_usd"]].values

        features_tensor = torch.FloatTensor(features)
        targets_tensor = torch.FloatTensor(targets)
        features_mean = features_tensor.mean(dim=0)
        features_std = features_tensor.std(dim=0) + 1e-8
        scaled_features = (features_tensor - features_mean) / features_std
        targets_min, targets_max = targets_tensor.min(), targets_tensor.max()
        scaled_targets = (targets_tensor - targets_min) / (targets_max - targets_min + 1e-8)

        X, y = [], []
        for i in range(len(merged) - sequence_length):
            X.append(scaled_features[i : i + sequence_length])
            y.append(scaled_targets[i + sequence_length])
        return np.array(X), np.array(y)

    def get_all_granularity_features(self, sequence_length: int = 24) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        output = {}
        for gran in self.granularities:
            try:
                X, y = self.prepare_features(granularity=gran, sequence_length=sequence_length)
                output[gran] = (X, y)
            except Exception as e:
                print(f"Could not prepare features for {gran}: {e}")
        return output

    def get_base_dataset(self, granularity: str = "1h") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get base dataset for training with features and targets.
        
        Args:
            granularity: Time granularity to use
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        price_data = self.load_price_data().get(granularity)
        chain_tvl = self.load_chain_tvl()
        santiment = self.load_santiment()

        if price_data is None or chain_tvl is None:
            raise ValueError(f"Price and chain TVL data required for {granularity}")

        price_data = price_data.set_index("timestamp").sort_index()
        freq = pd.infer_freq(price_data.index[:100]) or granularity
        chain_hourly = chain_tvl.set_index("timestamp").resample(freq).ffill()
        if santiment is not None:
            santiment_hourly = santiment.set_index("timestamp").resample(freq).ffill()
        else:
            santiment_hourly = pd.DataFrame(index=chain_hourly.index)

        merged = (
            price_data
            .join(chain_hourly[["tvl_usd"]], how="left")
            .join(santiment_hourly, how="left")
            .sort_index()
        )
        merged.fillna(method="ffill", inplace=True)

        # Compute all features
        merged["price_return"] = np.log(merged["close"]).diff().fillna(0)
        merged["tvl_change"] = merged["tvl_usd"].pct_change().fillna(0)
        merged["price_tvl_ratio"] = merged["close"] / merged["tvl_usd"]
        merged["return_vol_24"] = merged["price_return"].rolling(24).std().fillna(0)
        merged["volume_tvl_ratio"] = merged["volume"] / merged["tvl_usd"]

        merged["address_growth"] = merged.get("daily_active_addresses_value", pd.Series(0, index=merged.index)).pct_change().fillna(0)
        merged["social_dominance_change"] = merged.get("social_dominance_total_value", pd.Series(0, index=merged.index)).pct_change().fillna(0)
        merged["mcap_tvl_ratio"] = merged.get("marketcap_usd_value", pd.Series(0, index=merged.index)) / merged["tvl_usd"]

        close_d = self.find_optimal_d(merged["close"])
        merged["fracdiff_close"] = self.fracdiff(merged["close"], close_d)
        merged["return_entropy_24"] = self.compute_entropy(merged["price_return"], 24)
        merged["return_entropy_168"] = self.compute_entropy(merged["price_return"], 168)
        merged["cusum_flag"] = self.cusum_flag(merged["price_return"])
        merged["sadf_flag"] = self.sadf_flag(merged["close"])
        merged["vol_regime"] = self.volatility_regime(merged["price_return"])
        merged["return_vol_1"] = merged["price_return"].rolling(1).std().fillna(0)
        merged["return_vol_168"] = merged["price_return"].rolling(168).std().fillna(0)
        merged["parkinson_vol"] = self.parkinson_vol(merged["high"], merged["low"], 24).fillna(0)

        feature_cols = [
            "close", "volume", "tvl_usd", "price_return", "tvl_change", "price_tvl_ratio",
            "return_vol_24", "volume_tvl_ratio", "address_growth", "social_dominance_change",
            "mcap_tvl_ratio", "fracdiff_close", "return_entropy_24", "return_entropy_168",
            "cusum_flag", "sadf_flag", "vol_regime", "return_vol_1", "return_vol_168", "parkinson_vol"
        ]
        
        features_df = merged[feature_cols].fillna(0)
        targets_df = merged[["close", "tvl_usd"]].fillna(0)
        
        return features_df, targets_df
    
    def get_feature_cols(self) -> List[str]:
        """Get list of feature column names."""
        return [
            "close", "volume", "tvl_usd", "price_return", "tvl_change", "price_tvl_ratio",
            "return_vol_24", "volume_tvl_ratio", "address_growth", "social_dominance_change",
            "mcap_tvl_ratio", "fracdiff_close", "return_entropy_24", "return_entropy_168",
            "cusum_flag", "sadf_flag", "vol_regime", "return_vol_1", "return_vol_168", "parkinson_vol"
        ]

# --- Persistence Utilities ---
def save_numpy_arrays(X, y, out_dir: Path, prefix: str):
    np.save(out_dir / f"{prefix}_X.npy", X)
    np.save(out_dir / f"{prefix}_y.npy", y)

def save_pickle(obj, out_dir: Path, filename: str):
    with open(out_dir / filename, "wb") as f:
        pickle.dump(obj, f)

# --- Unified CLI ---
def main():
    parser = argparse.ArgumentParser(description="Unified Data Processing CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Bar sampling CLI
    bar_parser = subparsers.add_parser("bar_sample", help="Sample bars from CSV")
    bar_parser.add_argument("csv", type=Path)
    bar_parser.add_argument("--bar-type", default="tick")
    bar_parser.add_argument("--out", type=Path, required=True)

    # Simple feature generation CLI
    feat_parser = subparsers.add_parser("simple_features", help="Generate simple features from CSV")
    feat_parser.add_argument("csv", type=Path)
    feat_parser.add_argument("--out", type=Path, required=True)

    # Full feature engineering CLI
    full_parser = subparsers.add_parser("full_features", help="Generate full features and persist")
    full_parser.add_argument("--sequence-length", type=int, default=24)
    full_parser.add_argument("--data-dir", type=str, default="data")
    full_parser.add_argument("--out-dir", type=str, default="data/processed_features")

    args = parser.parse_args()

    if args.command == "bar_sample":
        df = pd.read_csv(
            args.csv,
            header=None,
            names=["timestamp", "open", "high", "low", "close", "volume"],
            usecols=range(6),
        )
        bars = sample_bars(df, args.bar_type)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(args.out)
        print(f"Bar-sampled data saved to {args.out}")

    elif args.command == "simple_features":
        df = pd.read_csv(args.csv)
        features = generate_features(df)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(args.out)
        print(f"Simple features saved to {args.out}")

    elif args.command == "full_features":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True)
        preprocessor = DataPreprocessor(data_dir=args.data_dir)
        all_granularity_data = preprocessor.get_all_granularity_features(sequence_length=args.sequence_length)
        for granularity, (X, y) in all_granularity_data.items():
            prefix = f"{granularity}_seq{args.sequence_length}"
            print(f"Saving features for {granularity} to {out_dir} as {prefix}_X.npy and {prefix}_y.npy")
            save_numpy_arrays(X, y, out_dir, prefix)
            meta = {
                "X_shape": X.shape,
                "y_shape": y.shape,
                "sequence_length": args.sequence_length,
                "granularity": granularity,
            }
            save_pickle(meta, out_dir, f"{prefix}_meta.pkl")
        print(f"Full features for all granularities saved to {out_dir}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 