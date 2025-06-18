import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path

# --- Model Classes (from hierarchical.py and simple.py) ---

class BaseModel(nn.Module):
    """Base model class with common functionality."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

class PriceLSTM(BaseModel):
    """LSTM model for price prediction."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class MetaMLP(BaseModel):
    """Meta-learning MLP for residual prediction."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        layers = []
        prev_size = input_size
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        self.mlp = nn.Sequential(*layers)
    def forward(self, base_pred: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([base_pred, features], dim=1)
        return self.mlp(x)

class ConfidenceGRU(BaseModel):
    """GRU model for confidence prediction."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])

# Simple models (from simple.py)
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

class SimpleMetaMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)

class SimpleConfidenceGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.fc(out[:, -1]))

# --- Model Factory and Info ---
def create_model(
    input_size: int,
    hidden_size: int,
    num_layers: int = 2,
    dropout: float = 0.2,
    model_type: str = "lstm"
) -> nn.Module:
    model_classes = {
        "lstm": PriceLSTM,
        "meta_mlp": MetaMLP,
        "confidence_gru": ConfidenceGRU
    }
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_classes[model_type](
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

def get_model_info(model: nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "model_type": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_size": getattr(model, "input_size", None),
        "hidden_size": getattr(model, "hidden_size", None),
        "num_layers": getattr(model, "num_layers", None),
        "dropout": getattr(model, "dropout", None)
    }

# --- EnsemblePredictor and build_ensemble (from ensemble.py, with imports fixed) ---

warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """
    PyTorch-only ensemble predictor implementing López de Prado's methodology.
    Features:
    - Hierarchical model architecture (Level-0, Level-1, Level-2)
    - Triple-barrier labeling with meta-labeling
    - Purged cross-validation with embargo
    - Sample uniqueness weighting
    - Risk-adjusted performance metrics
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 sequence_length: int = 24):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.hierarchical_model = None
        self.level_0_model = None
        self.level_1_model = None
        self.level_2_model = None
        self.training_history = {}
        self.cv_results = {}
        self.performance_metrics = {}
    def _prepare_data(self, 
                     features_df: pd.DataFrame, 
                     targets_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from features.labeling import create_labels, create_training_labels
        combined_df = features_df.copy()
        combined_df['close'] = targets_df['close']
        labeled_df = create_labels(combined_df, price_col='close')
        features = features_df.values
        features_tensor = torch.FloatTensor(features)
        features_mean = features_tensor.mean(dim=0)
        features_std = features_tensor.std(dim=0) + 1e-8
        scaled_features = (features_tensor - features_mean) / features_std
        X_seq = []
        for i in range(len(scaled_features) - self.sequence_length):
            X_seq.append(scaled_features[i:i + self.sequence_length])
        X = torch.stack(X_seq)
        y_ret, y_dir, sample_weights = create_training_labels(labeled_df, self.sequence_length)
        return X, y_ret, y_dir, sample_weights
    def _purged_cross_validation(self,
                                X: torch.Tensor,
                                y_ret: torch.Tensor,
                                y_dir: torch.Tensor,
                                sample_weights: torch.Tensor,
                                n_splits: int = 5,
                                embargo_hours: int = 3) -> Dict[str, List[float]]:
        from src.training.trainer import PurgedTimeSeriesSplit, compute_metrics, hierarchical_training_pipeline
        cv_splitter = PurgedTimeSeriesSplit(n_splits=n_splits, embargo_hours=embargo_hours)
        splits = cv_splitter.split(X, y_ret)
        cv_scores = {
            'level_0_mae': [],
            'level_0_directional_acc': [],
            'level_1_mae': [],
            'level_1_directional_acc': [],
            'level_2_accuracy': [],
            'information_ratio': []
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            print(f"\nFold {fold_idx + 1}/{n_splits}")
            X_train = X[train_indices]
            X_val = X[val_indices]
            y_ret_train = y_ret[train_indices]
            y_ret_val = y_ret[val_indices]
            y_dir_train = y_dir[train_indices]
            y_dir_val = y_dir[val_indices]
            train_dataset_ret = TensorDataset(X_train, y_ret_train)
            val_dataset_ret = TensorDataset(X_val, y_ret_val)
            train_dataset_dir = TensorDataset(X_train, y_dir_train)
            val_dataset_dir = TensorDataset(X_val, y_dir_val)
            train_loader_ret = DataLoader(train_dataset_ret, batch_size=32, shuffle=True)
            val_loader_ret = DataLoader(val_dataset_ret, batch_size=32)
            train_loader_dir = DataLoader(train_dataset_dir, batch_size=32, shuffle=True)
            val_loader_dir = DataLoader(val_dataset_dir, batch_size=32)
            results = hierarchical_training_pipeline(
                X_train, y_ret_train, y_dir_train, sample_weights[train_indices],
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_size=32,
                device=device
            )
            level_0_model = results["level_0"]["model"]
            level_1_model = results["level_1"]["model"]
            level_2_model = results["level_2"]["model"]
            level_0_model.eval()
            with torch.no_grad():
                val_pred_0 = level_0_model(X_val.to(device))
                level_0_metrics = compute_metrics(y_ret_val, val_pred_0.cpu(), "regression")
            level_1_model.eval()
            with torch.no_grad():
                base_pred = level_0_model(X_val.to(device))
                last_features = X_val[:, -1, :].to(device)
                val_pred_1 = level_1_model(base_pred, last_features)
                level_1_metrics = compute_metrics(y_ret_val, val_pred_1.cpu(), "regression")
            level_2_model.eval()
            with torch.no_grad():
                confidence = level_2_model(X_val.to(device))
                confidence_targets = (y_dir_val != 0).float().unsqueeze(1)
                level_2_metrics = compute_metrics(confidence_targets, confidence.cpu(), "classification")
            cv_scores['level_0_mae'].append(level_0_metrics['mae'])
            cv_scores['level_0_directional_acc'].append(level_0_metrics['directional_accuracy'])
            cv_scores['level_1_mae'].append(level_1_metrics['mae'])
            cv_scores['level_1_directional_acc'].append(level_1_metrics['directional_accuracy'])
            cv_scores['level_2_accuracy'].append(level_2_metrics['accuracy'])
            ir = level_1_metrics.get('information_ratio', 0.0)
            cv_scores['information_ratio'].append(ir)
            print(f"Fold {fold_idx + 1} Results:")
            print(f"  Level-0 MAE: {level_0_metrics['mae']:.6f}")
            print(f"  Level-1 MAE: {level_1_metrics['mae']:.6f}")
            print(f"  Level-2 Accuracy: {level_2_metrics['accuracy']:.4f}")
            print(f"  Information Ratio: {ir:.4f}")
        return cv_scores
    def train(self,
              features_df: pd.DataFrame,
              targets_df: pd.DataFrame,
              num_epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              run_cv: bool = True) -> Dict[str, any]:
        from src.data.features_all import DataPreprocessor
        from src.training.trainer import hierarchical_training_pipeline
        print("=== Training Ensemble Predictor ===")
        X, y_ret, y_dir, sample_weights = self._prepare_data(features_df, targets_df)
        print(f"Data prepared:")
        print(f"  X shape: {X.shape}")
        print(f"  y_ret shape: {y_ret.shape}")
        print(f"  y_dir shape: {y_dir.shape}")
        print(f"  sample_weights shape: {sample_weights.shape}")
        if run_cv:
            print("\n=== Running Purged Cross-Validation ===")
            self.cv_results = self._purged_cross_validation(X, y_ret, y_dir, sample_weights)
            print("\n=== Cross-Validation Results ===")
            for metric, scores in self.cv_results.items():
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"{metric}: {mean_score:.6f} ± {std_score:.6f}")
        print("\n=== Training Final Model ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = hierarchical_training_pipeline(
            X, y_ret, y_dir, sample_weights,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_size=batch_size,
            device=device
        )
        self.level_0_model = results["level_0"]["model"]
        self.level_1_model = results["level_1"]["model"]
        self.level_2_model = results["level_2"]["model"]
        self.hierarchical_model = results["hierarchical"]["model"]
        self.training_history = results
        self._compute_performance_metrics(X, y_ret, y_dir)
        return results
    def _compute_performance_metrics(self,
                                   X: torch.Tensor,
                                   y_ret: torch.Tensor,
                                   y_dir: torch.Tensor) -> None:
        from features.labeling import kelly_position_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        split_idx = int(0.8 * len(X))
        X_test = X[split_idx:].to(device)
        y_ret_test = y_ret[split_idx:]
        y_dir_test = y_dir[split_idx:]
        self.hierarchical_model.eval()
        with torch.no_grad():
            refined_pred, confidence = self.hierarchical_model(X_test, return_confidence=True)
            base_pred = self.hierarchical_model.get_base_prediction(X_test)
            refined_pred_np = refined_pred.cpu().numpy().flatten()
            base_pred_np = base_pred.cpu().numpy().flatten()
            confidence_np = confidence.cpu().numpy().flatten()
            y_ret_np = y_ret_test.numpy().flatten()
            y_dir_np = y_dir_test.numpy().flatten()
            self.performance_metrics = {
                'base_mae': np.mean(np.abs(y_ret_np - base_pred_np)),
                'refined_mae': np.mean(np.abs(y_ret_np - refined_pred_np)),
                'base_rmse': np.sqrt(np.mean((y_ret_np - base_pred_np) ** 2)),
                'refined_rmse': np.sqrt((np.mean((y_ret_np - refined_pred_np) ** 2))),
                'base_directional_acc': np.mean(np.sign(y_ret_np) == np.sign(base_pred_np)),
                'refined_directional_acc': np.mean(np.sign(y_ret_np) == np.sign(refined_pred_np)),
                'base_ir': np.mean(base_pred_np) / (np.std(base_pred_np) + 1e-8),
                'refined_ir': np.mean(refined_pred_np) / (np.std(refined_pred_np) + 1e-8),
                'mean_confidence': np.mean(confidence_np),
                'confidence_accuracy': np.mean((confidence_np > 0.5) == (y_dir_np != 0)),
                'hit_ratio': np.mean(np.abs(y_ret_np) > 0.01),
            }
            expected_returns = pd.Series(refined_pred_np)
            volatility = pd.Series(np.abs(refined_pred_np))
            probabilities = pd.Series(confidence_np)
            kelly_positions = kelly_position_size(
                probabilities, expected_returns, volatility,
                max_leverage=1.0
            )
            positions = kelly_positions.values
            returns = y_ret_np
            pnl = positions[:-1] * returns[1:]
            self.performance_metrics.update({
                'kelly_sharpe': np.mean(pnl) / (np.std(pnl) + 1e-8) * np.sqrt(24 * 365),
                'kelly_max_dd': self._calculate_max_drawdown(np.cumsum(pnl)),
                'kelly_calmar': (np.mean(pnl) * 24 * 365) / (self._calculate_max_drawdown(np.cumsum(pnl)) + 1e-8)
            })
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1e-8)
        return np.abs(np.min(drawdown))
    def predict(self, 
                features: torch.Tensor,
                return_confidence: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.hierarchical_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        self.hierarchical_model.eval()
        device = next(self.hierarchical_model.parameters()).device
        features = features.to(device)
        with torch.no_grad():
            if return_confidence:
                predictions, confidence = self.hierarchical_model(features, return_confidence=True)
                return predictions.cpu(), confidence.cpu()
            else:
                predictions = self.hierarchical_model(features, return_confidence=False)
                return predictions.cpu(), None
    def save_models(self, save_dir: str = "models") -> None:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        if self.hierarchical_model is not None:
            torch.save(self.hierarchical_model.state_dict(), save_path / "ensemble_hierarchical.pt")
            torch.save(self.level_0_model.state_dict(), save_path / "ensemble_level_0.pt")
            torch.save(self.level_1_model.state_dict(), save_path / "ensemble_level_1.pt")
            torch.save(self.level_2_model.state_dict(), save_path / "ensemble_level_2.pt")
            print(f"Models saved to {save_path}")
        else:
            print("No models to save. Train the ensemble first.")
    def load_models(self, save_dir: str = "models") -> None:
        from src.training.trainer import HierarchicalPredictor
        save_path = Path(save_dir)
        self.hierarchical_model = HierarchicalPredictor(
            self.input_size, self.hidden_size, self.num_layers, self.dropout
        )
        self.level_0_model = create_model(self.input_size, self.hidden_size, self.num_layers, model_type="lstm")
        self.level_1_model = create_model(self.input_size + 1, self.hidden_size, model_type="meta_mlp")
        self.level_2_model = create_model(self.input_size, self.hidden_size, model_type="confidence_gru")
        try:
            self.hierarchical_model.load_state_dict(torch.load(save_path / "ensemble_hierarchical.pt"))
            self.level_0_model.load_state_dict(torch.load(save_path / "ensemble_level_0.pt"))
            self.level_1_model.load_state_dict(torch.load(save_path / "ensemble_level_1.pt"))
            self.level_2_model.load_state_dict(torch.load(save_path / "ensemble_level_2.pt"))
            print(f"Models loaded from {save_path}")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
    def get_performance_summary(self) -> Dict[str, float]:
        return self.performance_metrics
    def get_cv_summary(self) -> Dict[str, Dict[str, float]]:
        cv_summary = {}
        for metric, scores in self.cv_results.items():
            cv_summary[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        return cv_summary

def build_ensemble(sequence_length: int = 24, 
                  hidden_size: int = 64, 
                  num_layers: int = 2,
                  num_epochs: int = 50) -> EnsemblePredictor:
    from src.data.features_all import DataPreprocessor
    print("=== Building ETH Price Prediction Ensemble ===")
    preprocessor = DataPreprocessor()
    features_df, targets_df = preprocessor.get_base_dataset()
    print(f"Loaded data:")
    print(f"  Features: {features_df.shape}")
    print(f"  Targets: {targets_df.shape}")
    print(f"  Feature columns: {len(preprocessor.get_feature_cols())}")
    input_size = len(preprocessor.get_feature_cols())
    ensemble = EnsemblePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        sequence_length=sequence_length
    )
    results = ensemble.train(
        features_df=features_df,
        targets_df=targets_df,
        num_epochs=num_epochs,
        run_cv=True
    )
    ensemble.save_models()
    print("\n=== Final Performance Summary ===")
    performance = ensemble.get_performance_summary()
    for metric, value in performance.items():
        print(f"{metric}: {value:.6f}")
    print("\n=== Cross-Validation Summary ===")
    cv_summary = ensemble.get_cv_summary()
    for metric, stats in cv_summary.items():
        print(f"{metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print("\nEnsemble training completed successfully!")
    return ensemble

if __name__ == "__main__":
    ensemble = build_ensemble(
        sequence_length=24,
        hidden_size=64,
        num_layers=2,
        num_epochs=50
    )
