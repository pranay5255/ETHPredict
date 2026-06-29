"""Lighter-only OHLCV feature engineering.

The active data path intentionally reads only public Lighter candle exports from
``data/raw/*-lighter-*.csv``. Multi-source TVL, Santiment, and Binance-specific
feature code is archived under ``archive/legacy_data_sources``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.data.lighter_client import RESOLUTION_TO_MS

warnings.filterwarnings("ignore")

DEFAULT_GRANULARITY = "5m"
MS_PER_HOUR = 60 * 60 * 1000

OHLCV_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

LIGHTER_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "price_return",
    "log_return",
    "volume_change",
    "quote_volume_change",
    "dollar_volume",
    "high_low_range",
    "close_open_return",
    "return_vol_1bar",
    "return_vol_24h",
    "return_vol_7d",
    "volume_zscore_24h",
    "fracdiff_close",
    "return_entropy_24h",
    "return_entropy_7d",
    "cusum_flag",
    "sadf_flag",
    "vol_regime",
    "parkinson_vol",
]


def bars_for_duration(granularity: str, *, hours: float) -> int:
    """Return the number of bars needed to preserve a wall-clock duration."""

    if granularity not in RESOLUTION_TO_MS:
        raise ValueError(f"Unsupported Lighter candle granularity: {granularity}")
    bars = round((hours * MS_PER_HOUR) / RESOLUTION_TO_MS[granularity])
    return max(1, int(bars))


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def sample_bars(df: pd.DataFrame, bar_type: str) -> pd.DataFrame:
    if bar_type == "tick":
        threshold = 1000
        metric = pd.Series(range(len(df)), index=df.index) + 1
    elif bar_type == "volume":
        threshold = 10000
        metric = df["volume"].cumsum()
    else:
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


def generate_features(df: pd.DataFrame, volatility_window: int = 288) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    df["return"] = df["close"].pct_change().fillna(0)
    df["rolling_vol"] = df["return"].rolling(volatility_window).std().fillna(0)
    return df


def save_features(asset: str, bar_type: str, df: pd.DataFrame) -> Path:
    date = pd.to_datetime(df["timestamp"].iloc[0], unit="ms").strftime("%Y%m%d")
    out_dir = Path("build/features") / asset / bar_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.parquet"
    df.to_parquet(out_path)
    return out_path


class DataPreprocessor:
    """Build model inputs from Lighter OHLCV CSV exports only."""

    def __init__(
        self,
        data_dir: str = "data",
        include_santiment: bool = False,
        granularities: Optional[Sequence[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.granularities = list(granularities or [DEFAULT_GRANULARITY])
        self.include_santiment = include_santiment

    def feature_window_bars(self, granularity: str) -> Dict[str, int]:
        return {
            "one_bar": 1,
            "one_hour": bars_for_duration(granularity, hours=1),
            "twenty_four_hours": bars_for_duration(granularity, hours=24),
            "seven_days": bars_for_duration(granularity, hours=24 * 7),
        }

    def load_price_data(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        raw_dir = self.data_dir / "raw"
        for granularity in self.granularities:
            candidates = sorted(raw_dir.glob(f"ETHUSDT-{granularity}-lighter-*.csv"))
            frames = []
            for csv_path in candidates:
                try:
                    df = pd.read_csv(csv_path, header=None, names=OHLCV_COLUMNS)
                    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
                    df = df.dropna(subset=["open_time"])
                    df["open_time"] = df["open_time"].astype("int64")
                    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
                    for column in OHLCV_COLUMNS:
                        df[column] = _clean_numeric(df[column])
                    frames.append(df)
                except Exception as exc:
                    print(f"Warning: Could not load {csv_path}: {exc}")
            if frames:
                price = pd.concat(frames, ignore_index=True)
                price = price.drop_duplicates(subset="timestamp").sort_values("timestamp")
                data[granularity] = price.reset_index(drop=True)
        return data

    def fracdiff(self, series: pd.Series, d: float, thres: float = 0.01) -> pd.Series:
        w = [1.0]
        for k in range(1, len(series)):
            w_ = -w[-1] * (d - k + 1) / k
            if abs(w_) < thres:
                break
            w.append(w_)
        weights = np.array(w[::-1])
        output = series.copy() * np.nan
        for idx in range(len(weights) - 1, len(series)):
            output.iloc[idx] = np.dot(weights, series.iloc[idx - len(weights) + 1 : idx + 1])
        return output

    def find_optimal_d(self, series: pd.Series, max_d: float = 1.0, corr_threshold: float = 0.97) -> float:
        best_d = 0.0
        clean = series.dropna()
        if len(clean) < 50:
            return best_d
        for d in np.arange(0.1, max_d + 0.01, 0.1):
            diffed = self.fracdiff(series, d).dropna()
            if len(diffed) < 25:
                continue
            try:
                _, pvalue, *_ = adfuller(diffed)
                corr = np.corrcoef(clean.iloc[-len(diffed) :], diffed)[0, 1]
                if pvalue < 0.05 and corr > corr_threshold:
                    best_d = float(d)
                    break
            except Exception:
                continue
        return best_d

    def compute_entropy(self, returns: pd.Series, window: int = 24) -> pd.Series:
        def shannon_entropy(values):
            counts, _ = np.histogram(values, bins=10)
            total = counts.sum()
            if total == 0:
                return 0.0
            probabilities = counts[counts > 0] / total
            return float(-np.sum(probabilities * np.log(probabilities)))

        return returns.rolling(window, min_periods=window).apply(shannon_entropy, raw=True)

    def cusum_flag(self, series: pd.Series, threshold: float = 2.0) -> pd.Series:
        std = series.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0, index=series.index)
        cusum = (series - series.mean()).cumsum()
        return (np.abs(cusum) > threshold * std).astype(int)

    def sadf_flag(self, series: pd.Series, max_window: int = 100, step: int = 12) -> pd.Series:
        flags = pd.Series(0, index=series.index)
        if len(series) < max_window:
            return flags
        for idx in range(max_window, len(series), step):
            window_data = series.iloc[idx - max_window + 1 : idx + 1]
            if window_data.isnull().any():
                continue
            try:
                adf_stat, *_ = adfuller(window_data)
            except Exception:
                continue
            end_idx = min(idx + step, len(series))
            flags.iloc[idx:end_idx] = int(adf_stat > -1.0)
        return flags

    def volatility_regime(self, returns: pd.Series, window: int = 24) -> pd.Series:
        vol = returns.rolling(window).std().fillna(0)
        low, high = vol.quantile(0.33), vol.quantile(0.67)
        regime = pd.Series(1, index=vol.index)
        regime[vol <= low] = 0
        regime[vol >= high] = 2
        return regime

    def parkinson_vol(self, high: pd.Series, low: pd.Series, window: int = 24) -> pd.Series:
        high = high.replace(0, np.nan)
        low = low.replace(0, np.nan)
        estimate = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(high / low) ** 2))
        return estimate.replace([np.inf, -np.inf], np.nan).rolling(window).mean()

    def _feature_frame(self, price_data: pd.DataFrame, granularity: str) -> pd.DataFrame:
        merged = price_data.set_index("timestamp").sort_index().copy()
        windows = self.feature_window_bars(granularity)
        window_24h = windows["twenty_four_hours"]
        window_7d = windows["seven_days"]
        step_1h = windows["one_hour"]

        for column in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
            merged[column] = _clean_numeric(merged[column])

        merged["price_return"] = _safe_pct_change(merged["close"])
        merged["log_return"] = np.log(merged["close"].replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).fillna(0)
        merged["volume_change"] = _safe_pct_change(merged["volume"])
        merged["quote_volume_change"] = _safe_pct_change(merged["quote_asset_volume"])
        merged["dollar_volume"] = merged["close"] * merged["volume"]
        merged["high_low_range"] = _safe_divide(merged["high"] - merged["low"], merged["close"])
        merged["close_open_return"] = _safe_divide(merged["close"] - merged["open"], merged["open"])
        merged["return_vol_1bar"] = merged["log_return"].rolling(windows["one_bar"]).std().fillna(0)
        merged["return_vol_24h"] = merged["log_return"].rolling(window_24h).std().fillna(0)
        merged["return_vol_7d"] = merged["log_return"].rolling(window_7d).std().fillna(0)

        volume_mean = merged["volume"].rolling(window_24h).mean()
        volume_std = merged["volume"].rolling(window_24h).std().replace(0, np.nan)
        merged["volume_zscore_24h"] = ((merged["volume"] - volume_mean) / volume_std).replace([np.inf, -np.inf], np.nan).fillna(0)

        close_d = self.find_optimal_d(merged["close"])
        merged["fracdiff_close"] = self.fracdiff(merged["close"], close_d)
        merged["return_entropy_24h"] = self.compute_entropy(merged["log_return"], window_24h)
        merged["return_entropy_7d"] = self.compute_entropy(merged["log_return"], window_7d)
        merged["cusum_flag"] = self.cusum_flag(merged["log_return"])
        merged["sadf_flag"] = self.sadf_flag(merged["close"], max_window=window_24h, step=step_1h)
        merged["vol_regime"] = self.volatility_regime(merged["log_return"], window=window_24h)
        merged["parkinson_vol"] = self.parkinson_vol(merged["high"], merged["low"], window_24h)

        return merged.replace([np.inf, -np.inf], np.nan).fillna(0)

    def _dataset_for_granularity(self, granularity: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        price_data = self.load_price_data().get(granularity)
        if price_data is None or price_data.empty:
            raise ValueError(f"Lighter OHLCV data required for {granularity}")

        feature_frame = self._feature_frame(price_data, granularity)
        features_df = feature_frame[self.get_feature_cols()].fillna(0)
        targets_df = feature_frame[["close", "volume"]].fillna(0)
        return features_df, targets_df

    def prepare_features(self, granularity: str, sequence_length: int = 288) -> Tuple[np.ndarray, np.ndarray]:
        features_df, targets_df = self._dataset_for_granularity(granularity)
        if len(features_df) <= sequence_length:
            raise ValueError(
                f"Need more than {sequence_length} Lighter OHLCV rows for {granularity}; got {len(features_df)}"
            )

        features_array = features_df.to_numpy(dtype=float, copy=True)
        targets_array = targets_df[["close"]].to_numpy(dtype=float, copy=True)
        features_mean = features_array.mean(axis=0)
        features_std = features_array.std(axis=0) + 1e-8
        scaled_features = (features_array - features_mean) / features_std
        targets_min, targets_max = targets_array.min(), targets_array.max()
        scaled_targets = (targets_array - targets_min) / (targets_max - targets_min + 1e-8)

        X, y = [], []
        for idx in range(len(features_df) - sequence_length):
            X.append(scaled_features[idx : idx + sequence_length])
            y.append(scaled_targets[idx + sequence_length])
        return np.stack(X), np.stack(y)

    def get_all_granularity_features(self, sequence_length: int = 288) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        output = {}
        for granularity in self.granularities:
            try:
                output[granularity] = self.prepare_features(granularity=granularity, sequence_length=sequence_length)
            except Exception as exc:
                print(f"Could not prepare features for {granularity}: {exc}")
        return output

    def get_base_dataset(self, granularity: str = DEFAULT_GRANULARITY) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._dataset_for_granularity(granularity)

    def get_feature_cols(self) -> List[str]:
        return list(LIGHTER_FEATURE_COLUMNS)


def save_numpy_arrays(X, y, out_dir: Path, prefix: str):
    np.save(out_dir / f"{prefix}_X.npy", X)
    np.save(out_dir / f"{prefix}_y.npy", y)


def save_pickle(obj, out_dir: Path, filename: str):
    with open(out_dir / filename, "wb") as f:
        pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser(description="Lighter-only data processing CLI")
    subparsers = parser.add_subparsers(dest="command")

    bar_parser = subparsers.add_parser("bar_sample", help="Sample bars from an OHLCV CSV")
    bar_parser.add_argument("csv", type=Path)
    bar_parser.add_argument("--bar-type", default="tick")
    bar_parser.add_argument("--out", type=Path, required=True)

    feat_parser = subparsers.add_parser("simple_features", help="Generate simple features from a CSV")
    feat_parser.add_argument("csv", type=Path)
    feat_parser.add_argument("--out", type=Path, required=True)

    full_parser = subparsers.add_parser("full_features", help="Generate Lighter-only sequence features")
    full_parser.add_argument("--sequence-length", type=int, default=288)
    full_parser.add_argument("--granularity", action="append", help="Lighter candle granularity to process, e.g. 5m. Repeat for multiple granularities.")
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
        out_dir.mkdir(parents=True, exist_ok=True)
        preprocessor = DataPreprocessor(data_dir=args.data_dir, granularities=args.granularity)
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
                "feature_columns": preprocessor.get_feature_cols(),
                "feature_window_bars": preprocessor.feature_window_bars(granularity),
            }
            save_pickle(meta, out_dir, f"{prefix}_meta.pkl")
        print(f"Full features saved to {out_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
