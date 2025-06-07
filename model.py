import torch
import torch.nn as nn

class TVLPredictor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super(TVLPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output of the sequence
        last_output = lstm_out[:, -1, :]
        # Pass through fully connected layers
        predictions = self.fc(last_output)
        return predictions

def create_model(input_size: int = 1, hidden_size: int = 64, num_layers: int = 2) -> TVLPredictor:
    """Helper function to create and initialize the model."""
    model = TVLPredictor(input_size, hidden_size, num_layers)
    return model 