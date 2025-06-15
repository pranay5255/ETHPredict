"""Market making experiment runner."""

import typer
from pathlib import Path
from src.orchestrator.config import ConfigManager
from src.orchestrator.main import run_single, run_grid, generate_report

app = typer.Typer(help="Market-making experiment runner")

@app.command()
def run(config: Path, tag: str = ""):
    """Execute a single experiment configuration."""
    config_manager = ConfigManager()
    cfg = config_manager.load_config(config)
    run_single(cfg, tag)

@app.command()
def grid(config: Path, max_par: int = 1, mode: str = "grid", n_trials: int = None):
    """Expand ranges in YAML and run grid.
    
    Args:
        config: Path to config file
        max_par: Maximum parallel processes
        mode: One of "grid", "random", "bayesian"
        n_trials: Number of trials for random/bayesian modes
    """
    config_manager = ConfigManager()
    cfg = config_manager.load_config(config)
    trials = config_manager.get_trials(cfg, mode=mode, n_trials=n_trials)
    run_grid(trials, max_par)

@app.command()
def report(input_dir: Path, output: Path = Path("summary.html")):
    """Generate experiment report.
    
    Args:
        input_dir: Directory containing experiment results
        output: Output HTML file path
    """
    generate_report(input_dir, output)

if __name__ == "__main__":
    app()
