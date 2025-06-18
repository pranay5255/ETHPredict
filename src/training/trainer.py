import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import pandas as pd
import itertools

from src.models.model import build_ensemble, EnsemblePredictor, PriceLSTM, MetaMLP, ConfidenceGRU, create_model


class HierarchicalPredictor(nn.Module):
    """
    Hierarchical predictor combining Level-0 (base LSTM), Level-1 (meta MLP), 
    and Level-2 (confidence GRU) models following López de Prado's methodology.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Level-0: Base price prediction (LSTM)
        self.level_0 = PriceLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Level-1: Meta-learning refinement (MLP)
        # Input: base prediction + last timestep features
        self.level_1 = MetaMLP(
            input_size=input_size + 1,  # +1 for base prediction
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Level-2: Confidence estimation (GRU)
        self.level_2 = ConfidenceGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, return_confidence: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through hierarchical model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            return_confidence: If True, return both prediction and confidence
            
        Returns:
            If return_confidence=False: refined prediction tensor
            If return_confidence=True: tuple of (refined_prediction, confidence)
        """
        # Level-0: Base prediction
        base_pred = self.level_0(x)  # [batch_size, 1]
        
        # Level-1: Meta-refinement using base prediction + last timestep features
        last_features = x[:, -1, :]  # [batch_size, input_size]
        refined_pred = self.level_1(base_pred, last_features)  # [batch_size, 1]
        
        if return_confidence:
            # Level-2: Confidence estimation
            confidence = self.level_2(x)  # [batch_size, 1]
            return refined_pred, confidence
        else:
            return refined_pred
    
    def get_base_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Get base Level-0 prediction."""
        return self.level_0(x)
    
    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Get Level-2 confidence prediction."""
        return self.level_2(x)


class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purging and embargo to prevent data leakage.
    
    Implements the methodology from López de Prado's "Advances in Financial Machine Learning":
    - Purging: Remove samples that overlap with the test set
    - Embargo: Add buffer period after test set to prevent information leakage
    """
    
    def __init__(self, n_splits: int = 5, embargo_hours: int = 3, purge_hours: int = 1):
        self.n_splits = n_splits
        self.embargo_hours = embargo_hours
        self.purge_hours = purge_hours
    
    def split(self, X: torch.Tensor, y: torch.Tensor) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation splits with purging and embargo.
        
        Args:
            X: Feature tensor [n_samples, sequence_length, n_features]
            y: Target tensor [n_samples, ...]
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n_samples = len(X)
        splits = []
        
        # Calculate split sizes
        test_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Test set boundaries
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            
            # Purging: remove samples that might overlap with test set
            purge_start = max(0, test_start - self.purge_hours)
            purge_end = min(n_samples, test_end + self.purge_hours)
            
            # Training set: all samples except test and purged regions
            train_indices = []
            
            # Before purged region
            if purge_start > 0:
                train_indices.extend(range(0, purge_start))
            
            # After purged region + embargo
            embargo_end = min(n_samples, purge_end + self.embargo_hours)
            if embargo_end < n_samples:
                train_indices.extend(range(embargo_end, n_samples))
            
            # Validation set
            val_indices = list(range(test_start, test_end))
            
            # Ensure we have training samples
            if len(train_indices) > 0 and len(val_indices) > 0:
                splits.append((np.array(train_indices), np.array(val_indices)))
        
        return splits


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, task_type: str = "regression") -> Dict[str, float]:
    """
    Compute comprehensive performance metrics for regression or classification tasks.
    
    Args:
        y_true: True values tensor
        y_pred: Predicted values tensor
        task_type: "regression" or "classification"
        
    Returns:
        Dictionary of computed metrics
    """
    # Convert to numpy for metric calculations
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy().flatten()
    else:
        y_true_np = np.array(y_true).flatten()
        
    if isinstance(y_pred, torch.Tensor):
        y_pred_np = y_pred.detach().cpu().numpy().flatten()
    else:
        y_pred_np = np.array(y_pred).flatten()
    
    metrics = {}
    
    if task_type == "regression":
        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(y_true_np, y_pred_np)
        metrics['mse'] = mean_squared_error(y_true_np, y_pred_np)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Directional accuracy
        y_true_sign = np.sign(y_true_np)
        y_pred_sign = np.sign(y_pred_np)
        metrics['directional_accuracy'] = np.mean(y_true_sign == y_pred_sign)
        
        # R-squared
        ss_res = np.sum((y_true_np - y_pred_np) ** 2)
        ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Information Ratio (Sharpe-like ratio for predictions)
        if np.std(y_pred_np) > 1e-8:
            metrics['information_ratio'] = np.mean(y_pred_np) / np.std(y_pred_np)
        else:
            metrics['information_ratio'] = 0.0
        
        # Hit ratio (percentage of predictions with correct direction and magnitude > threshold)
        threshold = np.std(y_true_np) * 0.1  # 10% of volatility
        correct_direction = (y_true_sign == y_pred_sign)
        significant_magnitude = (np.abs(y_true_np) > threshold)
        metrics['hit_ratio'] = np.mean(correct_direction & significant_magnitude)
        
    elif task_type == "classification":
        # Convert to binary predictions if probabilities
        if np.all((y_pred_np >= 0) & (y_pred_np <= 1)):
            y_pred_binary = (y_pred_np > 0.5).astype(int)
        else:
            y_pred_binary = y_pred_np.astype(int)
        
        y_true_binary = y_true_np.astype(int)
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
        
        # Log loss for probability predictions
        if np.all((y_pred_np >= 0) & (y_pred_np <= 1)):
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred_np, epsilon, 1 - epsilon)
            metrics['log_loss'] = -np.mean(y_true_binary * np.log(y_pred_clipped) + 
                                         (1 - y_true_binary) * np.log(1 - y_pred_clipped))
    
    return metrics


def hierarchical_training_pipeline(
    X: torch.Tensor,
    y_ret: torch.Tensor,
    y_dir: torch.Tensor,
    sample_weights: torch.Tensor,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Complete hierarchical training pipeline for the three-level model.
    
    Args:
        X: Input features [n_samples, sequence_length, n_features]
        y_ret: Return targets [n_samples, 1]
        y_dir: Direction targets [n_samples] (for confidence model)
        sample_weights: Sample weights [n_samples]
        input_size: Number of input features
        hidden_size: Hidden dimension size
        num_layers: Number of layers
        dropout: Dropout rate
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: PyTorch device
        
    Returns:
        Dictionary containing trained models and training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move data to device
    X = X.to(device)
    y_ret = y_ret.to(device)
    y_dir = y_dir.to(device)
    sample_weights = sample_weights.to(device)
    
    print(f"Training on device: {device}")
    print(f"Data shapes - X: {X.shape}, y_ret: {y_ret.shape}, y_dir: {y_dir.shape}")
    
    results = {}
    
    # ================================
    # Level-0: Base LSTM Training
    # ================================
    print("\n=== Training Level-0 Model (Base LSTM) ===")
    
    level_0_model = create_model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        model_type="lstm"
    ).to(device)
    
    level_0_optimizer = optim.Adam(level_0_model.parameters(), lr=learning_rate)
    level_0_criterion = nn.MSELoss(reduction='none')
    
    # Create data loaders
    dataset = TensorDataset(X, y_ret, sample_weights)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    level_0_losses = []
    level_0_model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y, batch_w in dataloader:
            level_0_optimizer.zero_grad()
            
            # Forward pass
            pred = level_0_model(batch_X)
            loss = level_0_criterion(pred, batch_y)
            
            # Apply sample weights
            weighted_loss = (loss.squeeze() * batch_w).mean()
            
            # Backward pass
            weighted_loss.backward()
            level_0_optimizer.step()
            
            epoch_loss += weighted_loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        level_0_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Level-0 Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    results["level_0"] = {
        "model": level_0_model,
        "training_losses": level_0_losses,
        "final_loss": level_0_losses[-1]
    }
    
    # ================================
    # Level-1: Meta-MLP Training
    # ================================
    print("\n=== Training Level-1 Model (Meta-MLP) ===")
    
    level_1_model = create_model(
        input_size=input_size + 1,  # +1 for base prediction
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        model_type="meta_mlp"
    ).to(device)
    
    level_1_optimizer = optim.Adam(level_1_model.parameters(), lr=learning_rate)
    level_1_criterion = nn.MSELoss(reduction='none')
    
    # Generate base predictions for meta-learning
    level_0_model.eval()
    with torch.no_grad():
        base_predictions = level_0_model(X)
    
    # Create meta-learning dataset
    last_features = X[:, -1, :]  # [n_samples, input_size]
    meta_dataset = TensorDataset(base_predictions, last_features, y_ret, sample_weights)
    meta_dataloader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True)
    
    level_1_losses = []
    level_1_model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for base_pred, features, batch_y, batch_w in meta_dataloader:
            level_1_optimizer.zero_grad()
            
            # Forward pass
            refined_pred = level_1_model(base_pred, features)
            loss = level_1_criterion(refined_pred, batch_y)
            
            # Apply sample weights
            weighted_loss = (loss.squeeze() * batch_w).mean()
            
            # Backward pass
            weighted_loss.backward()
            level_1_optimizer.step()
            
            epoch_loss += weighted_loss.item()
        
        avg_loss = epoch_loss / len(meta_dataloader)
        level_1_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Level-1 Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    results["level_1"] = {
        "model": level_1_model,
        "training_losses": level_1_losses,
        "final_loss": level_1_losses[-1]
    }
    
    # ================================
    # Level-2: Confidence GRU Training
    # ================================
    print("\n=== Training Level-2 Model (Confidence GRU) ===")
    
    level_2_model = create_model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        model_type="confidence_gru"
    ).to(device)
    
    level_2_optimizer = optim.Adam(level_2_model.parameters(), lr=learning_rate)
    level_2_criterion = nn.BCELoss(reduction='none')
    
    # Create confidence targets (binary: hit barrier or not)
    confidence_targets = (y_dir != 0).float().unsqueeze(1)  # [n_samples, 1]
    
    confidence_dataset = TensorDataset(X, confidence_targets, sample_weights)
    confidence_dataloader = DataLoader(confidence_dataset, batch_size=batch_size, shuffle=True)
    
    level_2_losses = []
    level_2_model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y, batch_w in confidence_dataloader:
            level_2_optimizer.zero_grad()
            
            # Forward pass
            confidence = level_2_model(batch_X)
            loss = level_2_criterion(confidence, batch_y)
            
            # Apply sample weights
            weighted_loss = (loss.squeeze() * batch_w).mean()
            
            # Backward pass
            weighted_loss.backward()
            level_2_optimizer.step()
            
            epoch_loss += weighted_loss.item()
        
        avg_loss = epoch_loss / len(confidence_dataloader)
        level_2_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Level-2 Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    results["level_2"] = {
        "model": level_2_model,
        "training_losses": level_2_losses,
        "final_loss": level_2_losses[-1]
    }
    
    # ================================
    # Create Combined Hierarchical Model
    # ================================
    print("\n=== Creating Hierarchical Predictor ===")
    
    hierarchical_model = HierarchicalPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Copy trained weights
    hierarchical_model.level_0.load_state_dict(level_0_model.state_dict())
    hierarchical_model.level_1.load_state_dict(level_1_model.state_dict())
    hierarchical_model.level_2.load_state_dict(level_2_model.state_dict())
    
    results["hierarchical"] = {
        "model": hierarchical_model,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout
    }
    
    print("Hierarchical training pipeline completed successfully!")
    return results


def main():
    """Main training function using the ensemble pipeline."""
    torch.manual_seed(42)
    np.random.seed(42)
    print("=== ETH Price Prediction - Ensemble Training ===")

    # Build and train the ensemble (handles all data, features, labels, and training internally)
    ensemble = build_ensemble(
        sequence_length=24,
        hidden_size=64,
        num_layers=2,
        num_epochs=50
    )

    # Save models (handled by build_ensemble, but can be called again if needed)
    # ensemble.save_models()

    # Print final summary
    print("\n=== Training Summary ===")
    perf = ensemble.get_performance_summary()
    for k, v in perf.items():
        print(f"{k}: {v}")
    print("\n=== Cross-Validation Summary ===")
    cv = ensemble.get_cv_summary()
    for metric, stats in cv.items():
        print(f"{metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print("\nEnsemble training completed successfully!")

if __name__ == "__main__":
    main() 