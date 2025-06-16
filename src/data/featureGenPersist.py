import os
import numpy as np
import pickle
from pathlib import Path
from src.data.loader import DataPreprocessor

def save_numpy_arrays(X, y, out_dir: Path, prefix: str):
    """Save feature and target arrays to .npy files with a given prefix."""
    np.save(out_dir / f"{prefix}_X.npy", X)
    np.save(out_dir / f"{prefix}_y.npy", y)

def save_pickle(obj, out_dir: Path, filename: str):
    """Save arbitrary Python object using pickle."""
    with open(out_dir / filename, "wb") as f:
        pickle.dump(obj, f)

def main(sequence_length=24):
    data_dir = Path("data")
    processed_dir = data_dir / "processed_features"
    processed_dir.mkdir(exist_ok=True)

    preprocessor = DataPreprocessor(data_dir=str(data_dir))
    all_granularity_data = preprocessor.get_all_granularity_features(sequence_length=sequence_length)

    for granularity, (X, y) in all_granularity_data.items():
        prefix = f"{granularity}_seq{sequence_length}"
        print(f"Saving features for {granularity} to {processed_dir} as {prefix}_X.npy and {prefix}_y.npy")
        save_numpy_arrays(X, y, processed_dir, prefix)
        # Optionally save as pickle as well for metadata or shapes
        meta = {
            "X_shape": X.shape,
            "y_shape": y.shape,
            "sequence_length": sequence_length,
            "granularity": granularity,
        }
        save_pickle(meta, processed_dir, f"{prefix}_meta.pkl")

if __name__ == "__main__":
    main(sequence_length=24)
