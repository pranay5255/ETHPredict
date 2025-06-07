import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List

class DataPreprocessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all relevant data files."""
        data = {}
        
        # Load chain TVL data
        chain_tvl = pd.read_csv(self.data_dir / "defillama_eth_chain_tvl_2025_04-05.csv")
        chain_tvl["timestamp"] = pd.to_datetime(chain_tvl["timestamp"])
        data["chain_tvl"] = chain_tvl

        # Load Binance hourly price data (April & May 2025)
        price_frames: List[pd.DataFrame] = []
        for month in ["04", "05"]:
            csv_path = self.data_dir / "processed" / f"processed_ETHUSDT-1h-2025-{month}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["timestamp"] = pd.to_datetime(df["open_time"])
                price_frames.append(df)
        if price_frames:
            price = pd.concat(price_frames, ignore_index=True)
            data["price"] = price

        # Load Santiment metrics (daily)
        santiment_path = self.data_dir / "santiment_metrics_april_may_2025.csv"
        if santiment_path.exists():
            santiment = pd.read_csv(santiment_path)
            santiment["timestamp"] = pd.to_datetime(santiment["datetime"])
            data["santiment"] = santiment

        # Load protocol TVL data
        protocol_path = self.data_dir / "defillama_eth_top10_tvl_2025_04-05.parquet"
        if protocol_path.exists():
            protocol_tvl = pd.read_parquet(protocol_path)
            protocol_tvl["timestamp"] = pd.to_datetime(protocol_tvl["timestamp"])
            data["protocol_tvl"] = protocol_tvl

        # Load protocol snapshots
        snapshot_path = self.data_dir / "defillama_eth_top10_snapshots_2025_04-05.csv"
        if snapshot_path.exists():
            snapshots = pd.read_csv(snapshot_path)
            data["snapshots"] = snapshots

        return data
    
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

        # Feature engineering
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
        ]

        features = merged[feature_cols].fillna(0).values
        targets = merged[["tvl_usd"]].values

        scaled_features = self.feature_scaler.fit_transform(features)
        scaled_targets = self.target_scaler.fit_transform(targets)

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

        chain_hourly = chain_tvl.set_index("timestamp").resample("1H").ffill()
        if santiment is not None:
            santiment_hourly = santiment.set_index("timestamp").resample("1H").ffill()
        else:
            santiment_hourly = pd.DataFrame(index=chain_hourly.index)

        merged = (
            price.set_index("timestamp")
            .join(chain_hourly[["tvl_usd"]], how="left")
            .join(santiment_hourly, how="left")
            .sort_index()
        )

        merged.fillna(method="ffill", inplace=True)

        merged["price_return"] = np.log(merged["close"]).diff().fillna(0)
        merged["tvl_change"] = merged["tvl_usd"].pct_change().fillna(0)
        merged["price_tvl_ratio"] = merged["close"] / merged["tvl_usd"]
        merged["return_vol_24h"] = merged["price_return"].rolling(24).std().fillna(0)
        merged["volume_tvl_ratio"] = merged["volume"] / merged["tvl_usd"]
        merged["address_growth"] = merged["daily_active_addresses_value"].pct_change().fillna(0)
        merged["social_dominance_change"] = merged["social_dominance_total_value"].pct_change().fillna(0)
        merged["mcap_tvl_ratio"] = merged["marketcap_usd_value"] / merged["tvl_usd"]

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
        ]

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
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original scale."""
        return self.target_scaler.inverse_transform(scaled_data)

def get_data(sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to get processed data ready for training."""
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    X, y = preprocessor.prepare_features(data, sequence_length)
    return X, y 