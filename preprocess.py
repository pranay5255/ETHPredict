import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict

class DataPreprocessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.scaler = MinMaxScaler()
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all relevant data files."""
        data = {}
        
        # Load chain TVL data
        chain_tvl = pd.read_csv(self.data_dir / "defillama_eth_chain_tvl_2025_04-05.csv")
        chain_tvl['timestamp'] = pd.to_datetime(chain_tvl['timestamp'])
        data['chain_tvl'] = chain_tvl
        
        # Load protocol TVL data
        protocol_tvl = pd.read_parquet(self.data_dir / "defillama_eth_top10_tvl_2025_04-05.parquet")
        protocol_tvl['timestamp'] = pd.to_datetime(protocol_tvl['timestamp'])
        data['protocol_tvl'] = protocol_tvl
        
        # Load protocol snapshots
        snapshots = pd.read_csv(self.data_dir / "defillama_eth_top10_snapshots_2025_04-05.csv")
        data['snapshots'] = snapshots
        
        return data
    
    def prepare_features(self, data: Dict[str, pd.DataFrame], 
                        sequence_length: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for the model.
        Returns X (features) and y (targets) as numpy arrays.
        """
        # Get chain TVL data
        chain_tvl = data['chain_tvl'].sort_values('timestamp')
        
        # Create features
        features = chain_tvl[['tvl_usd']].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - sequence_length):
            X.append(scaled_features[i:(i + sequence_length)])
            y.append(scaled_features[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original scale."""
        return self.scaler.inverse_transform(scaled_data)

def get_data(sequence_length: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to get processed data ready for training."""
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    X, y = preprocessor.prepare_features(data, sequence_length)
    return X, y 