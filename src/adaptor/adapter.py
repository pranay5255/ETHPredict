"""Adapter layer to reuse ETHPredict pipeline components."""

from typing import Any, Tuple

import torch

from preprocess import DataPreprocessor, get_data
from label import create_training_labels, create_labels
from train import hierarchical_training_pipeline


def load_data(cfg: dict) -> Any:
    """Load data using ETHPredict preprocessor."""
    data_dir = cfg.get("data_dir", "data")
    dp = DataPreprocessor(data_dir)
    features, targets = dp.get_base_dataset()
    return features, targets


def make_features(cfg: dict, data: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    features, targets = data
    sequence_length = cfg.get("sequence_length", 24)
    X, _ = get_data(sequence_length)
    labeled = create_labels(features.reset_index(), price_col="close")
    y_ret, y_dir, _ = create_training_labels(labeled, sequence_length=sequence_length)
    X = torch.FloatTensor(X)
    y_ret = torch.FloatTensor(y_ret)
    return X, y_ret


def train_models(cfg: dict, X: torch.Tensor, y: torch.Tensor):
    input_size = X.shape[2]
    params = cfg.get("model", {}).get("level0", {}).get("params", {})
    hidden_size = params.get("max_depth", 64)
    res = hierarchical_training_pipeline(
        X,
        y,
        y > 0,  # dummy direction
        torch.ones(len(y)),
        input_size=input_size,
        hidden_size=hidden_size,
    )
    return {
        "level0_loss": res["level_0"]["train_losses"][-1]
    }
