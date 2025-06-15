import torch
import torch.nn as nn
from typing import Dict, Any, Optional

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

def create_model(
    input_size: int,
    hidden_size: int,
    num_layers: int = 2,
    dropout: float = 0.2,
    model_type: str = "lstm"
) -> nn.Module:
    """
    Create a model instance.
    
    Args:
        input_size: Input feature dimension
        hidden_size: Hidden layer size
        num_layers: Number of layers
        dropout: Dropout rate
        model_type: One of "lstm", "meta_mlp", "confidence_gru"
    
    Returns:
        Model instance
    """
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
    """Get model information and statistics."""
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

