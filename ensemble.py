import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path

from preprocess import DataPreprocessor
from model import HierarchicalPredictor, PriceLSTM, MetaMLP, ConfidenceGRU, create_model
from label import create_labels, create_training_labels, kelly_position_size
from train import PurgedTimeSeriesSplit, compute_metrics, hierarchical_training_pipeline

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
        """
        Initialize the ensemble predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            sequence_length: Time series sequence length
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        
        # Models
        self.hierarchical_model = None
        self.level_0_model = None
        self.level_1_model = None
        self.level_2_model = None
        
        # Training history
        self.training_history = {}
        self.cv_results = {}
        
        # Evaluation metrics
        self.performance_metrics = {}
        
    def _prepare_data(self, 
                     features_df: pd.DataFrame, 
                     targets_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for training with triple-barrier labels.
        
        Args:
            features_df: Feature DataFrame
            targets_df: Target DataFrame
            
        Returns:
            Tuple of (X, y_ret, y_dir, sample_weights)
        """
        # Create triple-barrier labels
        combined_df = features_df.copy()
        combined_df['close'] = targets_df['close']
        labeled_df = create_labels(combined_df, price_col='close')
        
        # Create feature sequences
        features = features_df.values
        features_tensor = torch.FloatTensor(features)
        
        # Normalize features using z-score
        features_mean = features_tensor.mean(dim=0)
        features_std = features_tensor.std(dim=0) + 1e-8
        scaled_features = (features_tensor - features_mean) / features_std
        
        # Create sequences
        X_seq = []
        for i in range(len(scaled_features) - self.sequence_length):
            X_seq.append(scaled_features[i:i + self.sequence_length])
        X = torch.stack(X_seq)
        
        # Get labels for sequences
        y_ret, y_dir, sample_weights = create_training_labels(labeled_df, self.sequence_length)
        
        return X, y_ret, y_dir, sample_weights
    
    def _purged_cross_validation(self,
                                X: torch.Tensor,
                                y_ret: torch.Tensor,
                                y_dir: torch.Tensor,
                                sample_weights: torch.Tensor,
                                n_splits: int = 5,
                                embargo_hours: int = 3) -> Dict[str, List[float]]:
        """
        Perform purged k-fold cross-validation.
        
        Args:
            X: Feature sequences
            y_ret: Return targets
            y_dir: Direction targets
            sample_weights: Sample weights
            n_splits: Number of CV folds
            embargo_hours: Embargo period in hours
            
        Returns:
            Cross-validation results
        """
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
            
            # Split data
            X_train = X[train_indices]
            X_val = X[val_indices]
            y_ret_train = y_ret[train_indices]
            y_ret_val = y_ret[val_indices]
            y_dir_train = y_dir[train_indices]
            y_dir_val = y_dir[val_indices]
            
            # Create data loaders
            train_dataset_ret = TensorDataset(X_train, y_ret_train)
            val_dataset_ret = TensorDataset(X_val, y_ret_val)
            train_dataset_dir = TensorDataset(X_train, y_dir_train)
            val_dataset_dir = TensorDataset(X_val, y_dir_val)
            
            train_loader_ret = DataLoader(train_dataset_ret, batch_size=32, shuffle=True)
            val_loader_ret = DataLoader(val_dataset_ret, batch_size=32)
            train_loader_dir = DataLoader(train_dataset_dir, batch_size=32, shuffle=True)
            val_loader_dir = DataLoader(val_dataset_dir, batch_size=32)
            
            # Train models for this fold
            results = hierarchical_training_pipeline(
                X_train, y_ret_train, y_dir_train, sample_weights[train_indices],
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_size=32,
                device=device
            )
            
            # Evaluate on validation set
            level_0_model = results["level_0"]["model"]
            level_1_model = results["level_1"]["model"]
            level_2_model = results["level_2"]["model"]
            
            # Level-0 evaluation
            level_0_model.eval()
            with torch.no_grad():
                val_pred_0 = level_0_model(X_val.to(device))
                level_0_metrics = compute_metrics(y_ret_val, val_pred_0.cpu(), "regression")
            
            # Level-1 evaluation
            level_1_model.eval()
            with torch.no_grad():
                base_pred = level_0_model(X_val.to(device))
                last_features = X_val[:, -1, :].to(device)
                val_pred_1 = level_1_model(base_pred, last_features)
                level_1_metrics = compute_metrics(y_ret_val, val_pred_1.cpu(), "regression")
            
            # Level-2 evaluation
            level_2_model.eval()
            with torch.no_grad():
                confidence = level_2_model(X_val.to(device))
                confidence_targets = (y_dir_val != 0).float().unsqueeze(1)
                level_2_metrics = compute_metrics(confidence_targets, confidence.cpu(), "classification")
            
            # Store fold results
            cv_scores['level_0_mae'].append(level_0_metrics['mae'])
            cv_scores['level_0_directional_acc'].append(level_0_metrics['directional_accuracy'])
            cv_scores['level_1_mae'].append(level_1_metrics['mae'])
            cv_scores['level_1_directional_acc'].append(level_1_metrics['directional_accuracy'])
            cv_scores['level_2_accuracy'].append(level_2_metrics['accuracy'])
            
            # Information Ratio
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
        """
        Train the ensemble model.
        
        Args:
            features_df: Feature DataFrame
            targets_df: Target DataFrame
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            run_cv: Whether to run cross-validation
            
        Returns:
            Training results
        """
        print("=== Training Ensemble Predictor ===")
        
        # Prepare data
        X, y_ret, y_dir, sample_weights = self._prepare_data(features_df, targets_df)
        
        print(f"Data prepared:")
        print(f"  X shape: {X.shape}")
        print(f"  y_ret shape: {y_ret.shape}")
        print(f"  y_dir shape: {y_dir.shape}")
        print(f"  sample_weights shape: {sample_weights.shape}")
        
        # Cross-validation
        if run_cv:
            print("\n=== Running Purged Cross-Validation ===")
            self.cv_results = self._purged_cross_validation(X, y_ret, y_dir, sample_weights)
            
            # Print CV summary
            print("\n=== Cross-Validation Results ===")
            for metric, scores in self.cv_results.items():
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"{metric}: {mean_score:.6f} ± {std_score:.6f}")
        
        # Train final model on full dataset
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
        
        # Store models
        self.level_0_model = results["level_0"]["model"]
        self.level_1_model = results["level_1"]["model"]
        self.level_2_model = results["level_2"]["model"]
        self.hierarchical_model = results["hierarchical"]["model"]
        
        # Store training history
        self.training_history = results
        
        # Compute final performance metrics
        self._compute_performance_metrics(X, y_ret, y_dir)
        
        return results
    
    def _compute_performance_metrics(self,
                                   X: torch.Tensor,
                                   y_ret: torch.Tensor,
                                   y_dir: torch.Tensor) -> None:
        """
        Compute comprehensive performance metrics.
        
        Args:
            X: Feature sequences
            y_ret: Return targets
            y_dir: Direction targets
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Split for final evaluation (use last 20% as test set)
        split_idx = int(0.8 * len(X))
        X_test = X[split_idx:].to(device)
        y_ret_test = y_ret[split_idx:]
        y_dir_test = y_dir[split_idx:]
        
        self.hierarchical_model.eval()
        with torch.no_grad():
            # Get predictions
            refined_pred, confidence = self.hierarchical_model(X_test, return_confidence=True)
            
            # Base model predictions
            base_pred = self.hierarchical_model.get_base_prediction(X_test)
            
            # Convert to numpy
            refined_pred_np = refined_pred.cpu().numpy().flatten()
            base_pred_np = base_pred.cpu().numpy().flatten()
            confidence_np = confidence.cpu().numpy().flatten()
            y_ret_np = y_ret_test.numpy().flatten()
            y_dir_np = y_dir_test.numpy().flatten()
            
            # Compute metrics
            self.performance_metrics = {
                # Regression metrics
                'base_mae': np.mean(np.abs(y_ret_np - base_pred_np)),
                'refined_mae': np.mean(np.abs(y_ret_np - refined_pred_np)),
                'base_rmse': np.sqrt(np.mean((y_ret_np - base_pred_np) ** 2)),
                'refined_rmse': np.sqrt(np.mean((y_ret_np - refined_pred_np) ** 2)),
                
                # Directional accuracy
                'base_directional_acc': np.mean(np.sign(y_ret_np) == np.sign(base_pred_np)),
                'refined_directional_acc': np.mean(np.sign(y_ret_np) == np.sign(refined_pred_np)),
                
                # Information ratios
                'base_ir': np.mean(base_pred_np) / (np.std(base_pred_np) + 1e-8),
                'refined_ir': np.mean(refined_pred_np) / (np.std(refined_pred_np) + 1e-8),
                
                # Confidence metrics
                'mean_confidence': np.mean(confidence_np),
                'confidence_accuracy': np.mean((confidence_np > 0.5) == (y_dir_np != 0)),
                
                # Hit ratios
                'hit_ratio': np.mean(np.abs(y_ret_np) > 0.01),  # % of returns > 1%
            }
            
            # Kelly position sizing simulation
            expected_returns = pd.Series(refined_pred_np)
            volatility = pd.Series(np.abs(refined_pred_np))  # Simplified vol estimate
            probabilities = pd.Series(confidence_np)
            
            kelly_positions = kelly_position_size(
                probabilities, expected_returns, volatility,
                max_leverage=1.0
            )
            
            # Simulated P&L
            positions = kelly_positions.values
            returns = y_ret_np
            pnl = positions[:-1] * returns[1:]  # Next period returns
            
            self.performance_metrics.update({
                'kelly_sharpe': np.mean(pnl) / (np.std(pnl) + 1e-8) * np.sqrt(24 * 365),  # Annualized
                'kelly_max_dd': self._calculate_max_drawdown(np.cumsum(pnl)),
                'kelly_calmar': (np.mean(pnl) * 24 * 365) / (self._calculate_max_drawdown(np.cumsum(pnl)) + 1e-8)
            })
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1e-8)
        return np.abs(np.min(drawdown))
    
    def predict(self, 
                features: torch.Tensor,
                return_confidence: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Make predictions using the trained ensemble.
        
        Args:
            features: Input features [batch, seq_len, features]
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions and optionally confidence scores
        """
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
        """Save all trained models."""
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
        """Load pre-trained models."""
        save_path = Path(save_dir)
        
        # Initialize models
        self.hierarchical_model = HierarchicalPredictor(
            self.input_size, self.hidden_size, self.num_layers, self.dropout
        )
        self.level_0_model = create_model(self.input_size, self.hidden_size, self.num_layers, model_type="lstm")
        self.level_1_model = create_model(self.input_size + 1, self.hidden_size, model_type="meta_mlp")
        self.level_2_model = create_model(self.input_size, self.hidden_size, model_type="confidence_gru")
        
        # Load state dicts
        try:
            self.hierarchical_model.load_state_dict(torch.load(save_path / "ensemble_hierarchical.pt"))
            self.level_0_model.load_state_dict(torch.load(save_path / "ensemble_level_0.pt"))
            self.level_1_model.load_state_dict(torch.load(save_path / "ensemble_level_1.pt"))
            self.level_2_model.load_state_dict(torch.load(save_path / "ensemble_level_2.pt"))
            print(f"Models loaded from {save_path}")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance metrics summary."""
        return self.performance_metrics
    
    def get_cv_summary(self) -> Dict[str, Dict[str, float]]:
        """Get cross-validation summary."""
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
    """
    Build and train the ensemble predictor.
    
    Args:
        sequence_length: Time series sequence length
        hidden_size: Hidden layer size
        num_layers: Number of LSTM layers
        num_epochs: Number of training epochs
        
    Returns:
        Trained ensemble predictor
    """
    print("=== Building ETH Price Prediction Ensemble ===")
    
    # Load data
    preprocessor = DataPreprocessor()
    features_df, targets_df = preprocessor.get_base_dataset()
    
    print(f"Loaded data:")
    print(f"  Features: {features_df.shape}")
    print(f"  Targets: {targets_df.shape}")
    print(f"  Feature columns: {len(preprocessor.get_feature_cols())}")
    
    # Initialize ensemble
    input_size = len(preprocessor.get_feature_cols())
    ensemble = EnsemblePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        sequence_length=sequence_length
    )
    
    # Train ensemble
    results = ensemble.train(
        features_df=features_df,
        targets_df=targets_df,
        num_epochs=num_epochs,
        run_cv=True
    )
    
    # Save models
    ensemble.save_models()
    
    # Print results
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
