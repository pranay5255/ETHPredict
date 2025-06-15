import torch
import torch.nn as nn
from src.experiment_orchestrator import ExperimentOrchestrator

# Example model
class SimpleModel(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def run_experiment(params: dict, gpu_id: Optional[int] = None) -> dict:
    """
    Example experiment function that trains a simple model.
    
    Args:
        params: Dictionary of hyperparameters
        gpu_id: Optional GPU ID to use
    
    Returns:
        Dictionary of metrics
    """
    # Set device
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")
    
    # Create model
    model = SimpleModel(
        hidden_size=params["hidden_size"],
        dropout=params["dropout"]
    ).to(device)
    
    # Create dummy data
    x = torch.randn(100, 10).to(device)
    y = torch.randn(100, 1).to(device)
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.MSELoss()
    
    for epoch in range(params["epochs"]):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    # Return metrics
    return {
        "loss": loss.item(),
        "final_output_mean": output.mean().item()
    }

def main():
    # Define search space
    search_space = {
        "learning_rate": {
            "type": "continuous",
            "range": [0.0001, 0.1]
        },
        "hidden_size": {
            "type": "discrete",
            "values": [32, 64, 128, 256]
        },
        "dropout": {
            "type": "continuous",
            "range": [0.1, 0.5]
        },
        "epochs": {
            "type": "discrete",
            "values": [10, 20, 30]
        }
    }
    
    # Create orchestrator
    orchestrator = ExperimentOrchestrator(
        search_space=search_space,
        search_type="bayesian",  # Try "grid" or "random" for other search types
        n_trials=20,
        n_workers=2,
        gpu_ids=[0] if torch.cuda.is_available() else None,
        metric_name="loss",
        maximize=False
    )
    
    # Run experiments
    results = orchestrator.run_experiment(run_experiment)
    
    # Print summary
    summary = orchestrator.get_summary()
    print("\nExperiment Summary:")
    print(f"Total trials: {summary['total_trials']}")
    print(f"Successful trials: {summary['successful_trials']}")
    print(f"Failed trials: {summary['failed_trials']}")
    print("\nBest trial:")
    print(f"Parameters: {summary['best_trial']['params']}")
    print(f"Metrics: {summary['best_trial']['metrics']}")
    print("\nAverage metrics:")
    for metric, value in summary['average_metrics'].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 