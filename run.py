import yaml
import typer
from pathlib import Path
from src.orchestrator.main import run_single, run_grid, generate_report

app = typer.Typer(help="Market-making experiment runner")

@app.command()
def run(config: Path, tag: str = ""):
    """Execute a single experiment configuration."""
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    run_single(cfg, tag)

@app.command()
def grid(config: Path, max_par: int = 1):
    """Expand ranges in YAML and run grid."""
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    run_grid(cfg, max_par)

@app.command()
def report(input: Path, out: Path = Path("summary.html")):
    """Aggregate CSV results into HTML dashboard."""
    generate_report(input, out)

if __name__ == "__main__":
    app()
