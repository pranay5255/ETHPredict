import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PriceLSTM(nn.Module):
    """
    Level-0 (Base): Direct regression of next-hour price/TVL.
    Architecture: 2 × LSTM → FC layers
    Input: [batch, 24, F] where F = len(feature_cols)
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2):
        super(PriceLSTM, self).__init__()
        
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
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length=24, input_size=F]
        Returns:
            predictions: [batch_size, 1]
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last output of the sequence
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Pass through fully connected layers
        predictions = self.fc(last_output)  # [batch_size, 1]
        
        return predictions


class MetaMLP(nn.Module):
    """
    Level-1 (Meta-forecast): Combine base predictions with raw features to improve calibration.
    Architecture: 3 × FC layers with ReLU and Dropout
    Input: concat([pred_0, last_features]) → [batch, F+1]
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2):
        super(MetaMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, base_pred: torch.Tensor, last_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_pred: [batch_size, 1] - Level-0 predictions
            last_features: [batch_size, F] - Last timestep features
        Returns:
            refined_pred: [batch_size, 1] - Refined predictions
        """
        # Concatenate base prediction with last features
        x = torch.cat([base_pred, last_features], dim=1)  # [batch_size, F+1]
        
        # Pass through MLP
        refined_pred = self.network(x)  # [batch_size, 1]
        
        return refined_pred


class ConfidenceGRU(nn.Module):
    """
    Level-2 (Meta-label): Binary classifier for "confidence" of regression hit.
    Architecture: 1 × GRU → FC + sigmoid
    Input: Same [batch, 24, F] as Level-0
    Output: Confidence probability [0, 1]
    """
    
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.2):
        super(ConfidenceGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0  # Single layer, no dropout in GRU
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
        
        # Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length=24, input_size=F]
        Returns:
            confidence: [batch_size, 1] - Probability that the regression hit is trustworthy
        """
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Take the last output of the sequence
        last_output = gru_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Pass through classifier
        confidence = self.classifier(last_output)  # [batch_size, 1]
        
        return confidence


class HierarchicalPredictor(nn.Module):
    """
    Complete hierarchical model combining all three levels.
    
    Training sequence:
    1. Train Level-0 on (X_seq, y_ret) with MSELoss + Adam
    2. Freeze Level-0, generate predictions; train Level-1 on residuals (SmoothL1Loss)
    3. Train Level-2 with BCEWithLogitsLoss on direction labels
    
    Inference: pred_0 → pred_1 gives refined point estimate; conf = Level-2(x) gates position size
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2):
        super(HierarchicalPredictor, self).__init__()
        
        self.level_0 = PriceLSTM(input_size, hidden_size, num_layers, dropout)
        self.level_1 = MetaMLP(input_size + 1, hidden_size, dropout)  # +1 for base prediction
        self.level_2 = ConfidenceGRU(input_size, hidden_size // 2, dropout)
        
    def forward(self, x: torch.Tensor, return_confidence: bool = False) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length=24, input_size=F]
            return_confidence: Whether to return confidence scores
            
        Returns:
            If return_confidence=False: refined_predictions [batch_size, 1]
            If return_confidence=True: (refined_predictions, confidence) tuple
        """
        # Level-0: Base prediction
        base_pred = self.level_0(x)  # [batch_size, 1]
        
        # Level-1: Meta-forecast (refined prediction)
        last_features = x[:, -1, :]  # [batch_size, input_size]
        refined_pred = self.level_1(base_pred, last_features)  # [batch_size, 1]
        
        if return_confidence:
            # Level-2: Confidence estimation
            confidence = self.level_2(x)  # [batch_size, 1]
            return refined_pred, confidence
        else:
            return refined_pred
    
    def get_base_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Get Level-0 (base) predictions only."""
        return self.level_0(x)
    
    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Get Level-2 (confidence) predictions only."""
        return self.level_2(x)


def create_model(input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                dropout: float = 0.2, model_type: str = "hierarchical") -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_layers: Number of LSTM layers (for PriceLSTM)
        dropout: Dropout rate
        model_type: Type of model to create
        
    Returns:
        PyTorch model
    """
    if model_type == "hierarchical":
        return HierarchicalPredictor(input_size, hidden_size, num_layers, dropout)
    elif model_type == "lstm":
        return PriceLSTM(input_size, hidden_size, num_layers, dropout)
    elif model_type == "meta_mlp":
        return MetaMLP(input_size, hidden_size, dropout)
    elif model_type == "confidence_gru":
        return ConfidenceGRU(input_size, hidden_size // 2, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model: nn.Module) -> dict:
    """
    Get model information for monitoring and debugging.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_type": type(model).__name__,
        "device": next(model.parameters()).device.type if list(model.parameters()) else "cpu"
    }

