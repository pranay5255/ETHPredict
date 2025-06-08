import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from typing import Tuple, Dict, List

from preprocess import DataPreprocessor, get_data
from model import create_model, HierarchicalPredictor, PriceLSTM, MetaMLP, ConfidenceGRU, get_model_info
from label import create_training_labels, create_labels

warnings.filterwarnings('ignore')


class PurgedTimeSeriesSplit:
    """
    Purged K-Fold Cross-Validation for time series with embargo period.
    
    Based on López de Prado's methodology to prevent information leakage.
    """
    
    def __init__(self, n_splits: int = 5, embargo_hours: int = 3):
        self.n_splits = n_splits
        self.embargo_hours = embargo_hours
        
    def split(self, X: torch.Tensor, y: torch.Tensor) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/validation splits.
        
        Args:
            X: Features tensor [N, seq_len, features]
            y: Labels tensor [N, 1]
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # Validation set
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            # Training set with embargo
            train_indices = []
            
            # Before validation (with embargo)
            train_end_pre = max(0, val_start - self.embargo_hours)
            if train_end_pre > 0:
                train_indices.extend(range(0, train_end_pre))
            
            # After validation (with embargo)
            train_start_post = min(n_samples, val_end + self.embargo_hours)
            if train_start_post < n_samples:
                train_indices.extend(range(train_start_post, n_samples))
            
            val_indices = list(range(val_start, val_end))
            
            splits.append((np.array(train_indices), np.array(val_indices)))
            
        return splits


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, 
                   task_type: str = "regression") -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        task_type: "regression" or "classification"
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if task_type == "regression":
        # Convert to numpy
        y_true_np = y_true.detach().cpu().numpy().flatten()
        y_pred_np = y_pred.detach().cpu().numpy().flatten()
        
        # Regression metrics
        mse = np.mean((y_true_np - y_pred_np) ** 2)
        mae = np.mean(np.abs(y_true_np - y_pred_np))
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        direction_true = np.sign(y_true_np)
        direction_pred = np.sign(y_pred_np)
        directional_accuracy = np.mean(direction_true == direction_pred)
        
        # Information Ratio (simplified)
        if np.std(y_pred_np) > 0:
            ir = np.mean(y_pred_np) / np.std(y_pred_np)
        else:
            ir = 0.0
            
        metrics.update({
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "directional_accuracy": directional_accuracy,
            "information_ratio": ir
        })
        
    elif task_type == "classification":
        # Convert to numpy and binarize
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = (y_pred.detach().cpu().numpy() > 0.5).astype(int)
        
        accuracy = accuracy_score(y_true_np, y_pred_np)
        precision = precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
        recall = recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
        f1 = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
        
        metrics.update({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
    
    return metrics


def train_level_0(
    model: PriceLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: torch.device = None
) -> Tuple[List[float], List[float], Dict[str, float]]:
    """
    Train Level-0 (Base) model on regression task.
    
    Args:
        model: PriceLSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Tuple of (train_losses, val_losses, final_metrics)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_val_true = []
        all_val_pred = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                all_val_true.append(y_batch)
                all_val_pred.append(outputs)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Level-0 Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}')
    
    # Compute final metrics
    all_val_true = torch.cat(all_val_true)
    all_val_pred = torch.cat(all_val_pred)
    final_metrics = compute_metrics(all_val_true, all_val_pred, "regression")
    
    return train_losses, val_losses, final_metrics


def train_level_1(
    level_0_model: PriceLSTM,
    level_1_model: MetaMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    device: torch.device = None
) -> Tuple[List[float], List[float], Dict[str, float]]:
    """
    Train Level-1 (Meta-forecast) model on residuals.
    
    Args:
        level_0_model: Frozen Level-0 model
        level_1_model: MetaMLP model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Tuple of (train_losses, val_losses, final_metrics)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    level_0_model = level_0_model.to(device)
    level_1_model = level_1_model.to(device)
    
    # Freeze Level-0
    level_0_model.eval()
    for param in level_0_model.parameters():
        param.requires_grad = False
    
    criterion = nn.SmoothL1Loss()  # More robust to outliers
    optimizer = optim.Adam(level_1_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        level_1_model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Get Level-0 predictions
            with torch.no_grad():
                base_pred = level_0_model(X_batch)
            
            # Get last timestep features
            last_features = X_batch[:, -1, :]
            
            optimizer.zero_grad()
            
            # Level-1 prediction
            meta_pred = level_1_model(base_pred, last_features)
            
            # Loss on residuals (meta_pred should improve upon base_pred)
            loss = criterion(meta_pred, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(level_1_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        level_1_model.eval()
        val_loss = 0
        all_val_true = []
        all_val_pred = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                base_pred = level_0_model(X_batch)
                last_features = X_batch[:, -1, :]
                meta_pred = level_1_model(base_pred, last_features)
                
                loss = criterion(meta_pred, y_batch)
                val_loss += loss.item()
                
                all_val_true.append(y_batch)
                all_val_pred.append(meta_pred)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Level-1 Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}')
    
    # Compute final metrics
    all_val_true = torch.cat(all_val_true)
    all_val_pred = torch.cat(all_val_pred)
    final_metrics = compute_metrics(all_val_true, all_val_pred, "regression")
    
    return train_losses, val_losses, final_metrics


def train_level_2(
    model: ConfidenceGRU,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    device: torch.device = None
) -> Tuple[List[float], List[float], Dict[str, float]]:
    """
    Train Level-2 (Confidence) model on direction classification.
    
    Args:
        model: ConfidenceGRU model
        train_loader: Training data loader with direction labels
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Tuple of (train_losses, val_losses, final_metrics)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    criterion = nn.BCELoss()  # Binary cross-entropy for confidence
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.float().to(device)
            
            optimizer.zero_grad()
            confidence = model(X_batch)
            
            # Convert direction labels to binary confidence targets
            # 1 if direction is not 0 (hit barrier), 0 if timeout
            confidence_target = (y_batch != 0).float().unsqueeze(1)
            
            loss = criterion(confidence, confidence_target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_val_true = []
        all_val_pred = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.float().to(device)
                
                confidence = model(X_batch)
                confidence_target = (y_batch != 0).float().unsqueeze(1)
                
                loss = criterion(confidence, confidence_target)
                val_loss += loss.item()
                
                all_val_true.append(confidence_target)
                all_val_pred.append(confidence)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Level-2 Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}')
    
    # Compute final metrics
    all_val_true = torch.cat(all_val_true)
    all_val_pred = torch.cat(all_val_pred)
    final_metrics = compute_metrics(all_val_true, all_val_pred, "classification")
    
    return train_losses, val_losses, final_metrics


def hierarchical_training_pipeline(
    X: torch.Tensor,
    y_ret: torch.Tensor,
    y_dir: torch.Tensor,
    sample_weights: torch.Tensor,
    input_size: int,
    sequence_length: int = 24,
    hidden_size: int = 64,
    num_layers: int = 2,
    batch_size: int = 32,
    device: torch.device = None
) -> Dict[str, any]:
    """
    Complete hierarchical training pipeline.
    
    Args:
        X: Feature sequences [N, seq_len, features]
        y_ret: Return targets [N, 1]
        y_dir: Direction targets [N]
        sample_weights: Sample weights [N]
        input_size: Number of input features
        sequence_length: Sequence length
        hidden_size: Hidden layer size
        num_layers: Number of LSTM layers
        batch_size: Batch size
        device: Training device
        
    Returns:
        Dictionary with trained models and metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on device: {device}")
    print(f"Data shape: X={X.shape}, y_ret={y_ret.shape}, y_dir={y_dir.shape}")
    
    # Split data (80-20)
    split_idx = int(0.8 * len(X))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_ret_train, y_ret_val = y_ret[:split_idx], y_ret[split_idx:]
    y_dir_train, y_dir_val = y_dir[:split_idx], y_dir[split_idx:]
    
    # Create data loaders
    train_dataset_ret = TensorDataset(X_train, y_ret_train)
    val_dataset_ret = TensorDataset(X_val, y_ret_val)
    train_dataset_dir = TensorDataset(X_train, y_dir_train)
    val_dataset_dir = TensorDataset(X_val, y_dir_val)
    
    train_loader_ret = DataLoader(train_dataset_ret, batch_size=batch_size, shuffle=True)
    val_loader_ret = DataLoader(val_dataset_ret, batch_size=batch_size)
    train_loader_dir = DataLoader(train_dataset_dir, batch_size=batch_size, shuffle=True)
    val_loader_dir = DataLoader(val_dataset_dir, batch_size=batch_size)
    
    results = {}
    
    # Step 1: Train Level-0 (Base LSTM)
    print("\n=== Training Level-0 (Base LSTM) ===")
    level_0_model = create_model(input_size, hidden_size, num_layers, model_type="lstm")
    level_0_train_losses, level_0_val_losses, level_0_metrics = train_level_0(
        level_0_model, train_loader_ret, val_loader_ret, device=device
    )
    
    results["level_0"] = {
        "model": level_0_model,
        "train_losses": level_0_train_losses,
        "val_losses": level_0_val_losses,
        "metrics": level_0_metrics
    }
    
    print(f"Level-0 Final Metrics: {level_0_metrics}")
    
    # Step 2: Train Level-1 (Meta-MLP)
    print("\n=== Training Level-1 (Meta-MLP) ===")
    level_1_model = create_model(input_size + 1, hidden_size, model_type="meta_mlp")
    level_1_train_losses, level_1_val_losses, level_1_metrics = train_level_1(
        level_0_model, level_1_model, train_loader_ret, val_loader_ret, device=device
    )
    
    results["level_1"] = {
        "model": level_1_model,
        "train_losses": level_1_train_losses,
        "val_losses": level_1_val_losses,
        "metrics": level_1_metrics
    }
    
    print(f"Level-1 Final Metrics: {level_1_metrics}")
    
    # Step 3: Train Level-2 (Confidence GRU)
    print("\n=== Training Level-2 (Confidence GRU) ===")
    level_2_model = create_model(input_size, hidden_size, model_type="confidence_gru")
    level_2_train_losses, level_2_val_losses, level_2_metrics = train_level_2(
        level_2_model, train_loader_dir, val_loader_dir, device=device
    )
    
    results["level_2"] = {
        "model": level_2_model,
        "train_losses": level_2_train_losses,
        "val_losses": level_2_val_losses,
        "metrics": level_2_metrics
    }
    
    print(f"Level-2 Final Metrics: {level_2_metrics}")
    
    # Step 4: Create complete hierarchical model
    print("\n=== Creating Hierarchical Model ===")
    hierarchical_model = HierarchicalPredictor(input_size, hidden_size, num_layers)
    hierarchical_model.level_0.load_state_dict(level_0_model.state_dict())
    hierarchical_model.level_1.load_state_dict(level_1_model.state_dict())
    hierarchical_model.level_2.load_state_dict(level_2_model.state_dict())
    
    results["hierarchical"] = {"model": hierarchical_model}
    
    return results


def main():
    """Main training function."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== ETH Price Prediction - Hierarchical Training ===")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    features_df, targets_df = preprocessor.get_base_dataset()
    
    # Create labels using triple-barrier method
    print("Creating triple-barrier labels...")
    labeled_df = create_labels(features_df.reset_index(), price_col="close")
    
    # Get feature sequences
    X, _ = get_data(sequence_length=24)
    X = torch.FloatTensor(X)
    
    # Get labels
    y_ret, y_dir, sample_weights = create_training_labels(labeled_df, sequence_length=24)
    
    print(f"Dataset shapes:")
    print(f"  Features: {X.shape}")
    print(f"  Returns: {y_ret.shape}")
    print(f"  Directions: {y_dir.shape}")
    print(f"  Weights: {sample_weights.shape}")
    
    # Training parameters
    input_size = X.shape[2]
    hidden_size = 64
    num_layers = 2
    batch_size = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run hierarchical training
    results = hierarchical_training_pipeline(
        X, y_ret, y_dir, sample_weights,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        device=device
    )
    
    # Save models
    print("\n=== Saving Models ===")
    Path("models").mkdir(exist_ok=True)
    
    torch.save(results["level_0"]["model"].state_dict(), "models/price_lstm.pt")
    torch.save(results["level_1"]["model"].state_dict(), "models/meta_mlp.pt")
    torch.save(results["level_2"]["model"].state_dict(), "models/conf_gru.pt")
    torch.save(results["hierarchical"]["model"].state_dict(), "models/hierarchical_model.pt")
    
    print("Models saved successfully!")
    
    # Plot training history
    print("\n=== Plotting Training History ===")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Level-0 losses
    axes[0, 0].plot(results["level_0"]["train_losses"], label='Train')
    axes[0, 0].plot(results["level_0"]["val_losses"], label='Validation')
    axes[0, 0].set_title('Level-0 (Base LSTM) Training')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Level-1 losses
    axes[0, 1].plot(results["level_1"]["train_losses"], label='Train')
    axes[0, 1].plot(results["level_1"]["val_losses"], label='Validation')
    axes[0, 1].set_title('Level-1 (Meta-MLP) Training')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Level-2 losses
    axes[1, 0].plot(results["level_2"]["train_losses"], label='Train')
    axes[1, 0].plot(results["level_2"]["val_losses"], label='Validation')
    axes[1, 0].set_title('Level-2 (Confidence GRU) Training')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Metrics comparison
    metrics_names = ['directional_accuracy', 'information_ratio', 'mae']
    level_0_values = [results["level_0"]["metrics"].get(m, 0) for m in metrics_names]
    level_1_values = [results["level_1"]["metrics"].get(m, 0) for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, level_0_values, width, label='Level-0')
    axes[1, 1].bar(x + width/2, level_1_values, width, label='Level-1')
    axes[1, 1].set_title('Model Performance Comparison')
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('hierarchical_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final summary
    print("\n=== Training Summary ===")
    print(f"Level-0 Directional Accuracy: {results['level_0']['metrics']['directional_accuracy']:.4f}")
    print(f"Level-1 Directional Accuracy: {results['level_1']['metrics']['directional_accuracy']:.4f}")
    print(f"Level-2 Classification Accuracy: {results['level_2']['metrics']['accuracy']:.4f}")
    
    print("\n=== Model Information ===")
    for level in ["level_0", "level_1", "level_2"]:
        model_info = get_model_info(results[level]["model"])
        print(f"{level}: {model_info}")
    
    print("\nHierarchical training completed successfully!")


if __name__ == "__main__":
    main() 