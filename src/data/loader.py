import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
from statsmodels.tsa.stattools import adfuller
from scipy.signal import welch
from scipy.stats import norm
import torch

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all relevant data files."""
        data = {}
        
        # Load chain TVL data
        chain_tvl = pd.read_csv(self.data_dir / "defillama_eth_chain_tvl_2025_04-05.csv")
        chain_tvl["timestamp"] = pd.to_datetime(chain_tvl["timestamp"])
        # Remove timezone info if present
        if chain_tvl["timestamp"].dt.tz is not None:
            chain_tvl["timestamp"] = chain_tvl["timestamp"].dt.tz_localize(None)
        data["chain_tvl"] = chain_tvl

        # Load Binance hourly price data
        price_frames: List[pd.DataFrame] = []
        binance_columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        for month in ["04", "05"]:
            csv_path = self.data_dir / "raw" / f"ETHUSDT-1h-2025-{month}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, header=None, names=binance_columns)
                    try:
                        df["timestamp"] = pd.to_datetime(df["open_time"], unit='ms')
                    except (ValueError, TypeError, OSError):
                        df["timestamp"] = pd.to_datetime(df["open_time"])
                    # Ensure timestamp is timezone-naive
                    if df["timestamp"].dt.tz is not None:
                        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
                    price_frames.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {csv_path}: {e}")
                    continue
        
        if price_frames:
            price = pd.concat(price_frames, ignore_index=True)
            data["price"] = price

        # Load Santiment metrics
        santiment_path = self.data_dir / "santiment_metrics_april_may_2025.csv"
        if santiment_path.exists():
            santiment = pd.read_csv(santiment_path)
            santiment["timestamp"] = pd.to_datetime(santiment["datetime"])
            # Ensure timestamp is timezone-naive
            if santiment["timestamp"].dt.tz is not None:
                santiment["timestamp"] = santiment["timestamp"].dt.tz_localize(None)
            data["santiment"] = santiment

        return data
    
    def fracdiff(self, series: pd.Series, d: float, thres: float = 0.01) -> pd.Series:
        """
        Fractional differentiation as per López de Prado.
        
        Args:
            series: Time series to differentiate
            d: Fractional differentiation order
            thres: Threshold for truncating weights
            
        Returns:
            Fractionally differentiated series
        """
        # Compute weights
        w = [1.0]
        for k in range(1, len(series)):
            w.append(-w[-1] * (d - k + 1) / k)
            if abs(w[-1]) < thres:
                w = w[:-1]
                break
        
        # Apply fractional differentiation
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(w), len(series)):
            result.iloc[i] = np.dot(w, series.iloc[i-len(w)+1:i+1])
        
        return result.fillna(0)
    
    def find_optimal_d(self, series: pd.Series, max_d: float = 1.0, 
                      corr_threshold: float = 0.97) -> float:
        """
        Find optimal fractional differentiation order.
        
        Args:
            series: Time series to analyze
            max_d: Maximum d to test
            corr_threshold: Minimum correlation with original series
            
        Returns:
            Optimal fractional differentiation order
        """
        best_d = 0.0
        
        for d in np.arange(0.1, max_d + 0.1, 0.1):
            fracdiff_series = self.fracdiff(series, d)
            
            # Check stationarity
            try:
                adf_stat, p_value, _, _, _, _ = adfuller(fracdiff_series.dropna())
                if p_value < 0.05:  # Stationary at 95% confidence
                    # Check correlation with original
                    corr = np.corrcoef(series.dropna(), fracdiff_series.dropna())[0, 1]
                    if corr >= corr_threshold:
                        best_d = d
                        break
            except:
                continue
                
        return best_d
    
    def compute_entropy(self, returns: pd.Series, window: int = 24) -> pd.Series:
        """
        Compute rolling Shannon entropy of returns.
        
        Args:
            returns: Return series
            window: Rolling window size
            
        Returns:
            Entropy series
        """
        def shannon_entropy(x):
            # Discretize returns into bins
            hist, _ = np.histogram(x, bins=10, density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            return -np.sum(hist * np.log(hist))
        
        return returns.rolling(window).apply(shannon_entropy, raw=True)
    
    def cusum_flag(self, series: pd.Series, threshold: float = 2.0) -> pd.Series:
        """
        CUSUM structural break detection.
        
        Args:
            series: Time series
            threshold: Threshold for flagging breaks
            
        Returns:
            Binary flag series
        """
        mean_val = series.mean()
        cusum_pos = (series - mean_val).cumsum()
        cusum_neg = (series - mean_val).cumsum()
        
        # Flag when CUSUM exceeds threshold
        flags = (np.abs(cusum_pos) > threshold * series.std()) | \
                (np.abs(cusum_neg) > threshold * series.std())
        
        return flags.astype(int)
    
    def sadf_flag(self, series: pd.Series, min_window: int = 24) -> pd.Series:
        """
        Simplified SADF (Supremum Augmented Dickey-Fuller) explosive move detection.
        
        Args:
            series: Time series
            min_window: Minimum window for ADF test
            
        Returns:
            Binary flag series
        """
        flags = pd.Series(0, index=series.index)
        
        for i in range(min_window, len(series)):
            window_data = series.iloc[max(0, i-min_window):i+1]
            try:
                adf_stat, _, _, _, _, _ = adfuller(window_data)
                # Flag explosive behavior (very negative ADF stat indicates mean reversion)
                if adf_stat > -1.0:  # Weak mean reversion suggests explosive behavior
                    flags.iloc[i] = 1
            except:
                continue
                
        return flags
    
    def volatility_regime(self, returns: pd.Series, window: int = 24) -> pd.Series:
        """
        Discretize volatility into regimes.
        
        Args:
            returns: Return series
            window: Rolling window for volatility calculation
            
        Returns:
            Regime series (0=low, 1=mid, 2=high)
        """
        vol = returns.rolling(window).std()
        
        # Use percentiles to define regimes
        low_thresh = vol.quantile(0.33)
        high_thresh = vol.quantile(0.67)
        
        regime = pd.Series(1, index=vol.index)  # Default to mid
        regime[vol <= low_thresh] = 0  # Low volatility
        regime[vol >= high_thresh] = 2  # High volatility
        
        return regime

    def prepare_features(self, data: Dict[str, pd.DataFrame],
                        sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature and target sequences for the LSTM model."""

        # Merge available data on a common hourly timestamp
        price = data.get("price")
        chain_tvl = data.get("chain_tvl")
        santiment = data.get("santiment")

        if price is None or chain_tvl is None:
            raise ValueError("Price and chain TVL data are required")

        # Resample chain TVL and Santiment metrics to hourly frequency
        chain_hourly = chain_tvl.set_index("timestamp").resample("1H").ffill()
        if santiment is not None:
            santiment_hourly = santiment.set_index("timestamp").resample("1H").ffill()
        else:
            santiment_hourly = pd.DataFrame(index=chain_hourly.index)

        # Merge datasets
        merged = (
            price.set_index("timestamp")
            .join(chain_hourly[["tvl_usd"]], how="left")
            .join(santiment_hourly, how="left")
            .sort_index()
        )

        # Forward fill any remaining gaps
        merged.fillna(method="ffill", inplace=True)

        # Basic feature engineering
        merged["price_return"] = np.log(merged["close"]).diff().fillna(0)
        merged["tvl_change"] = merged["tvl_usd"].pct_change().fillna(0)
        merged["price_tvl_ratio"] = merged["close"] / merged["tvl_usd"]

        # Additional engineered features for ensemble models
        merged["return_vol_24h"] = (
            merged["price_return"].rolling(24).std().fillna(0)
        )
        merged["volume_tvl_ratio"] = merged["volume"] / merged["tvl_usd"]
        merged["address_growth"] = (
            merged["daily_active_addresses_value"].pct_change().fillna(0)
        )
        merged["social_dominance_change"] = (
            merged["social_dominance_total_value"].pct_change().fillna(0)
        )
        merged["mcap_tvl_ratio"] = merged["marketcap_usd_value"] / merged["tvl_usd"]

        # Advanced features from signal.md
        # Fractional differentiation
        close_d = self.find_optimal_d(merged["close"])
        merged["fracdiff_close"] = self.fracdiff(merged["close"], close_d)
        
        # Entropy features
        merged["return_entropy_24h"] = self.compute_entropy(merged["price_return"], 24)
        merged["return_entropy_168h"] = self.compute_entropy(merged["price_return"], 168)  # 7 days
        
        # Structural break flags
        merged["cusum_flag"] = self.cusum_flag(merged["price_return"])
        merged["sadf_flag"] = self.sadf_flag(merged["close"])
        
        # Volatility regimes
        merged["vol_regime"] = self.volatility_regime(merged["price_return"])
        
        # Additional volatility measures
        merged["return_vol_1h"] = merged["price_return"].rolling(1).std().fillna(0)
        merged["return_vol_168h"] = merged["price_return"].rolling(168).std().fillna(0)
        
        # High-low estimator (Parkinson volatility)
        merged["parkinson_vol"] = np.sqrt(
            0.25 * np.log(2) * (np.log(merged["high"] / merged["low"]) ** 2)
        ).rolling(24).mean().fillna(0)

        # Feature column order (CRITICAL: maintain this exact order)
        feature_cols = [
            # Original base features
            "close",
            "volume",
            "tvl_usd",
            "daily_active_addresses_value",
            "dev_activity_value",
            "social_volume_total_value",
            "network_growth_value",
            "price_return",
            "tvl_change",
            "price_tvl_ratio",
            "return_vol_24h",
            "volume_tvl_ratio",
            "address_growth",
            "social_dominance_change",
            "mcap_tvl_ratio",
            # Advanced features from signal.md
            "fracdiff_close",
            "return_entropy_24h",
            "return_entropy_168h",
            "cusum_flag",
            "sadf_flag",
            "vol_regime",
            "return_vol_1h",
            "return_vol_168h",
            "parkinson_vol",
        ]

        features = merged[feature_cols].fillna(0).values
        targets = merged[["tvl_usd"]].values

        # Convert to tensors and normalize
        features_tensor = torch.FloatTensor(features)
        targets_tensor = torch.FloatTensor(targets)
        
        # Normalize features using z-score
        features_mean = features_tensor.mean(dim=0)
        features_std = features_tensor.std(dim=0) + 1e-8
        scaled_features = (features_tensor - features_mean) / features_std
        
        # Normalize targets using min-max scaling
        targets_min = targets_tensor.min()
        targets_max = targets_tensor.max()
        scaled_targets = (targets_tensor - targets_min) / (targets_max - targets_min + 1e-8)

        X, y = [], []
        for i in range(len(merged) - sequence_length):
            X.append(scaled_features[i : i + sequence_length])
            y.append(scaled_targets[i + sequence_length])

        return np.array(X), np.array(y)

    def get_base_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return feature and target DataFrames for ensemble models."""
        data = self.load_data()

        price = data.get("price")
        chain_tvl = data.get("chain_tvl")
        santiment = data.get("santiment")

        if price is None or chain_tvl is None:
            raise ValueError("Price and chain TVL data are required")

        # Ensure all timestamps are timezone-naive before setting as index
        for df in [price, chain_tvl, santiment]:
            if df is not None and df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # Set indices and resample
        chain_hourly = chain_tvl.set_index("timestamp").resample("1H").ffill()
        if santiment is not None:
            santiment_hourly = santiment.set_index("timestamp").resample("1H").ffill()
        else:
            santiment_hourly = pd.DataFrame(index=chain_hourly.index)

        # Join datasets
        merged = (
            price.set_index("timestamp")
            .join(chain_hourly[["tvl_usd"]], how="left")
            .join(santiment_hourly, how="left")
            .sort_index()
        )

        merged.fillna(method="ffill", inplace=True)

        # Basic features
        merged["price_return"] = np.log(merged["close"]).diff().fillna(0)
        merged["tvl_change"] = merged["tvl_usd"].pct_change().fillna(0)
        merged["price_tvl_ratio"] = merged["close"] / merged["tvl_usd"]
        merged["return_vol_24h"] = merged["price_return"].rolling(24).std().fillna(0)
        merged["volume_tvl_ratio"] = merged["volume"] / merged["tvl_usd"]
        merged["address_growth"] = merged["daily_active_addresses_value"].pct_change().fillna(0)
        merged["social_dominance_change"] = merged["social_dominance_total_value"].pct_change().fillna(0)
        merged["mcap_tvl_ratio"] = merged["marketcap_usd_value"] / merged["tvl_usd"]

        # Advanced features from signal.md
        close_d = self.find_optimal_d(merged["close"])
        merged["fracdiff_close"] = self.fracdiff(merged["close"], close_d)
        merged["return_entropy_24h"] = self.compute_entropy(merged["price_return"], 24)
        merged["return_entropy_168h"] = self.compute_entropy(merged["price_return"], 168)
        merged["cusum_flag"] = self.cusum_flag(merged["price_return"])
        merged["sadf_flag"] = self.sadf_flag(merged["close"])
        merged["vol_regime"] = self.volatility_regime(merged["price_return"])
        merged["return_vol_1h"] = merged["price_return"].rolling(1).std().fillna(0)
        merged["return_vol_168h"] = merged["price_return"].rolling(168).std().fillna(0)
        merged["parkinson_vol"] = np.sqrt(
            0.25 * np.log(2) * (np.log(merged["high"] / merged["low"]) ** 2)
        ).rolling(24).mean().fillna(0)

        # EXACT same feature column order as prepare_features
        feature_cols = [
            "close",
            "volume",
            "tvl_usd",
            "daily_active_addresses_value",
            "dev_activity_value",
            "social_volume_total_value",
            "network_growth_value",
            "price_return",
            "tvl_change",
            "price_tvl_ratio",
            "return_vol_24h",
            "volume_tvl_ratio",
            "address_growth",
            "social_dominance_change",
            "mcap_tvl_ratio",
            "fracdiff_close",
            "return_entropy_24h",
            "return_entropy_168h",
            "cusum_flag",
            "sadf_flag",
            "vol_regime",
            "return_vol_1h",
            "return_vol_168h",
            "parkinson_vol",
        ]

        # Targets for different models
        target_cols = [
            "tvl_usd",
            "close",
            "return_vol_24h",
            "daily_active_addresses_value",
            "social_volume_total_value",
        ]

        features_df = merged[feature_cols].iloc[:-1].fillna(0)
        targets_df = merged[target_cols].shift(-1).iloc[:-1].fillna(0)

        return features_df, targets_df
    
    def get_feature_cols(self) -> List[str]:
        """Return the exact feature column order."""
        return [
            "close",
            "volume", 
            "tvl_usd",
            "daily_active_addresses_value",
            "dev_activity_value",
            "social_volume_total_value",
            "network_growth_value",
            "price_return",
            "tvl_change",
            "price_tvl_ratio",
            "return_vol_24h",
            "volume_tvl_ratio",
            "address_growth",
            "social_dominance_change",
            "mcap_tvl_ratio",
            "fracdiff_close",
            "return_entropy_24h",
            "return_entropy_168h",
            "cusum_flag",
            "sadf_flag",
            "vol_regime",
            "return_vol_1h",
            "return_vol_168h",
            "parkinson_vol",
        ]

    def inverse_transform(self, scaled_data: np.ndarray, 
                         targets_min: float, targets_max: float) -> np.ndarray:
        """Convert scaled predictions back to original scale."""
        return scaled_data * (targets_max - targets_min) + targets_min

    def _ensure_timezone_naive(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Ensure timestamp column is timezone-naive."""
        if df[timestamp_col].dt.tz is not None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)
        return df

def get_data(sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to get processed data ready for training."""
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    X, y = preprocessor.prepare_features(data, sequence_length)
    return X, y 