#!/usr/bin/env python3
"""
ETHPredict CLI Runner

This is the main command-line interface for the ETHPredict system.
It provides commands to run single experiments, parameter sweeps, and generate reports.

Usage:
    python runner.py run --config configs/training.yml
    python runner.py grid --training configs/training.yml --market-making configs/market_making.yml
    python runner.py report --input results/ --output report.html
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import typer
import yaml
import torch
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add project root to Python path
project_root = Path("configs/config.yml").parent
sys.path.insert(0, str(project_root))

# Import existing orchestrator and components
from src.experiment_orchestrator import ExperimentOrchestrator
try:
    from src.utils.logging import setup_logging
except ImportError:
    # Fallback logging setup if module doesn't exist
    def setup_logging(log_file=None):
        logging.basicConfig(level=logging.INFO)

# Initialize Typer app and Rich console
app = typer.Typer(
    name="ethpredict",
    help="ETHPredict CLI - Hierarchical ML model for ETH price prediction and market making",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()

# Global configuration - simplified to one comprehensive config
DEFAULT_CONFIG = "configs/config.yml"

def load_config(config_file: str = None) -> Dict[str, Any]:
    """Load configuration file."""
    config_path = config_file or DEFAULT_CONFIG
    
    if not Path(config_path).exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        console.print("[yellow]Available configs in configs/:[/yellow]")
        for cfg in Path("configs").glob("*.yml"):
            console.print(f"  - {cfg}")
        raise typer.Exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        console.print(f"[green]✓[/green] Loaded config: {config_path}")
    
    return config

def setup_experiment_directory(experiment_id: str) -> Path:
    """Create experiment directory and setup logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("results") / f"{experiment_id}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = exp_dir / "experiment.log"
    setup_logging(log_file=str(log_file))
    
    return exp_dir

def create_search_space_from_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Create parameter search space from configuration."""
    search_space = {}
    
    # Extract parameter ranges from backtest optimization section
    if "optimization" in config and "parameter_ranges" in config["optimization"]:
        param_ranges = config["optimization"]["parameter_ranges"]
        
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                # Determine if it's continuous or discrete
                if isinstance(param_range[0], float) or isinstance(param_range[1], float):
                    search_space[param_name] = {
                        "type": "continuous",
                        "range": param_range
                    }
                else:
                    # Create discrete values between min and max
                    values = list(range(param_range[0], param_range[1] + 1))
                    search_space[param_name] = {
                        "type": "discrete", 
                        "values": values
                    }
    
    # Add some default parameters if search space is empty
    if not search_space:
        search_space = {
            "learning_rate": {"type": "continuous", "range": [0.0001, 0.1]},
            "gamma": {"type": "continuous", "range": [0.1, 2.0]},
            "inventory_limit": {"type": "discrete", "values": [1000, 5000, 10000, 20000]},
            "quote_spread": {"type": "continuous", "range": [0.0005, 0.01]}
        }
    
    return search_space

def create_experiment_function(config: Dict[str, Any]):
    """Create experiment function that integrates with the orchestrator."""
    
    def experiment_function(params: Dict[str, Any], gpu_id: Optional[int] = None) -> Dict[str, float]:
        """
        Main experiment function that trains model and runs backtest.
        
        Args:
            params: Dictionary of hyperparameters
            gpu_id: Optional GPU ID to use
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Set device
            device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")
            
            # Update config with parameters
            experiment_config = config.copy()
            
            # Update model parameters
            if "learning_rate" in params:
                experiment_config.setdefault("training", {})["learning_rate"] = params["learning_rate"]
            
            if "gamma" in params:
                experiment_config.setdefault("market_maker", {})["gamma"] = params["gamma"]
            
            if "inventory_limit" in params:
                experiment_config.setdefault("market_maker", {})["inventory_limit"] = params["inventory_limit"]
                
            if "quote_spread" in params:
                experiment_config.setdefault("market_maker", {})["quote_spread"] = params["quote_spread"]
            
            # Import and run training pipeline
            from src.training.trainer import run_experiment as run_training
            from src.simulator.backtest import run_backtest
            
            # Run training
            training_metrics = run_training(experiment_config, gpu_id)
            
            # Run backtest
            backtest_metrics = run_backtest(experiment_config)
            
            # Combine metrics
            combined_metrics = {
                **training_metrics,
                **backtest_metrics
            }
            
            # Ensure required metrics are present
            if "sharpe_ratio" not in combined_metrics:
                combined_metrics["sharpe_ratio"] = combined_metrics.get("sharpe", 0.0)
            
            if "max_drawdown" not in combined_metrics:
                combined_metrics["max_drawdown"] = combined_metrics.get("drawdown", 0.0)
                
            return combined_metrics
            
        except Exception as e:
            logging.error(f"Experiment failed: {str(e)}")
            return {
                "sharpe_ratio": -999.0,
                "max_drawdown": 1.0,
                "error": str(e)
            }
    
    return experiment_function

@app.command()
def run(
    config_file: str = typer.Option(None, "--config", "-c", help="Configuration file"),
    experiment_id: str = typer.Option("single_run", "--id", help="Experiment ID"),
    tag: str = typer.Option("", "--tag", help="Experiment tag"),
    gpu_id: Optional[int] = typer.Option(None, "--gpu", help="GPU ID to use")
):
    """
    Run a single experiment with the given configuration.
    
    This command loads the configuration file, trains the model,
    and runs the backtesting pipeline.
    """
    console.print(Panel.fit("🚀 [bold blue]ETHPredict Single Run[/bold blue]", border_style="blue"))
    
    try:
        # Load configuration
        console.print("[yellow]Loading configuration...[/yellow]")
        config = load_config(config_file)
        
        # Setup experiment directory
        exp_dir = setup_experiment_directory(experiment_id)
        console.print(f"[green]✓[/green] Experiment directory: {exp_dir}")
        
        # Save merged configuration
        config_file = exp_dir / "config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        console.print(f"[green]✓[/green] Saved merged config: {config_file}")
        
        # Create experiment function
        experiment_fn = create_experiment_function(config)
        
        # Run single experiment
        console.print("[yellow]Starting experiment...[/yellow]")
        start_time = time.time()
        
        # Extract parameters from config for single run
        params = {
            "learning_rate": config.get("training", {}).get("learning_rate", 0.001),
            "gamma": config.get("market_maker", {}).get("gamma", 0.5),
            "inventory_limit": config.get("market_maker", {}).get("inventory_limit", 10000),
            "quote_spread": config.get("market_maker", {}).get("quote_spread", 0.001),
        }
        
        # Run experiment
        metrics = experiment_fn(params, gpu_id)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Display results
        console.print(f"[green]✓[/green] Experiment completed in {duration:.2f} seconds")
        
        # Create results table
        table = Table(title="Experiment Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))
        
        console.print(table)
        
        # Save results
        results_file = exp_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "experiment_id": experiment_id,
                "tag": tag,
                "config": config,
                "params": params,
                "metrics": metrics,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        console.print(f"[green]✓[/green] Results saved: {results_file}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Experiment failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def grid(
    config_file: str = typer.Option(None, "--config", "-c", help="Configuration file"),
    experiment_id: str = typer.Option("grid_search", "--id", help="Experiment ID"),
    n_trials: int = typer.Option(100, "--trials", "-n", help="Number of trials"),
    n_workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    search_mode: str = typer.Option("bayesian", "--mode", help="Search mode: grid, random, bayesian"),
    gpu_ids: str = typer.Option("0", "--gpus", help="Comma-separated GPU IDs to use")
):
    """
    Run parameter sweep (grid search, random search, or Bayesian optimization).
    
    This command performs hyperparameter optimization using the specified search strategy.
    """
    console.print(Panel.fit(f"🔍 [bold blue]ETHPredict Parameter Sweep ({search_mode})[/bold blue]", border_style="blue"))
    
    try:
        # Parse GPU IDs
        gpu_list = [int(x.strip()) for x in gpu_ids.split(",") if x.strip().isdigit()] if gpu_ids else None
        
        # Load configuration
        console.print("[yellow]Loading configuration...[/yellow]")
        config = load_config(config_file)
        
        # Setup experiment directory
        exp_dir = setup_experiment_directory(experiment_id)
        console.print(f"[green]✓[/green] Experiment directory: {exp_dir}")
        
        # Create search space
        search_space = create_search_space_from_config(config)
        console.print(f"[green]✓[/green] Created search space with {len(search_space)} parameters")
        
        # Display search space
        table = Table(title="Search Space", show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Range/Values", style="green")
        
        for param_name, param_config in search_space.items():
            param_type = param_config["type"]
            if param_type == "continuous":
                range_str = f"{param_config['range'][0]} - {param_config['range'][1]}"
            else:
                range_str = str(param_config["values"])
            table.add_row(param_name, param_type, range_str)
        
        console.print(table)
        
        # Create orchestrator
        orchestrator = ExperimentOrchestrator(
            search_space=search_space,
            search_type=search_mode,
            n_trials=n_trials,
            n_workers=n_workers,
            gpu_ids=gpu_list,
            checkpoint_dir=str(exp_dir / "checkpoints"),
            metric_name="sharpe_ratio",
            maximize=True
        )
        
        # Create experiment function
        experiment_fn = create_experiment_function(config)
        
        # Run parameter sweep
        console.print(f"[yellow]Starting {search_mode} search with {n_trials} trials...[/yellow]")
        start_time = time.time()
        
        with Progress() as progress:
            task = progress.add_task("[green]Running experiments...", total=n_trials)
            
            # Run experiments
            results = orchestrator.run_experiment(experiment_fn)
            
            progress.update(task, completed=n_trials)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get summary
        summary = orchestrator.get_summary()
        console.print(f"[green]✓[/green] Parameter sweep completed in {duration:.2f} seconds")
        
        # Display summary
        summary_table = Table(title="Sweep Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Trials", str(summary["total_trials"]))
        summary_table.add_row("Successful Trials", str(summary["successful_trials"]))
        summary_table.add_row("Failed Trials", str(summary["failed_trials"]))
        summary_table.add_row("Duration", f"{duration:.2f} seconds")
        
        console.print(summary_table)
        
        # Display best trial
        if summary["best_trial"]:
            best_trial = summary["best_trial"]
            console.print("\n[bold]Best Trial Results:[/bold]")
            
            best_table = Table(show_header=True, header_style="bold magenta")
            best_table.add_column("Parameter/Metric", style="cyan")
            best_table.add_column("Value", style="green")
            
            # Add parameters
            for param, value in best_trial["params"].items():
                if isinstance(value, float):
                    best_table.add_row(f"param_{param}", f"{value:.4f}")
                else:
                    best_table.add_row(f"param_{param}", str(value))
            
            # Add metrics
            for metric, value in best_trial["metrics"].items():
                if isinstance(value, float):
                    best_table.add_row(f"metric_{metric}", f"{value:.4f}")
                else:
                    best_table.add_row(f"metric_{metric}", str(value))
            
            console.print(best_table)
        
        # Save results
        results_file = exp_dir / "sweep_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "experiment_id": experiment_id,
                "search_mode": search_mode,
                "n_trials": n_trials,
                "config": config,
                "search_space": search_space,
                "summary": summary,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        console.print(f"[green]✓[/green] Results saved: {results_file}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Parameter sweep failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def report(
    input_dir: str = typer.Argument(..., help="Input directory containing experiment results"),
    output: str = typer.Option("report.html", "--output", "-o", help="Output report file"),
    format: str = typer.Option("html", "--format", "-f", help="Report format: html, pdf, json"),
    include_plots: bool = typer.Option(True, "--plots/--no-plots", help="Include performance plots")
):
    """
    Generate comprehensive experiment report from results.
    
    This command analyzes experiment results and generates a detailed report
    with performance metrics, visualizations, and statistical analysis.
    """
    console.print(Panel.fit("📊 [bold blue]ETHPredict Report Generation[/bold blue]", border_style="blue"))
    
    try:
        input_path = Path(input_dir)
        if not input_path.exists():
            console.print(f"[red]✗[/red] Input directory not found: {input_dir}")
            raise typer.Exit(1)
        
        console.print(f"[yellow]Analyzing results in: {input_path}[/yellow]")
        
        # Find all result files
        result_files = list(input_path.glob("**/*.json"))
        if not result_files:
            console.print(f"[red]✗[/red] No JSON result files found in {input_dir}")
            raise typer.Exit(1)
        
        console.print(f"[green]✓[/green] Found {len(result_files)} result files")
        
        # Load and combine results
        all_results = []
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    all_results.append(data)
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Skipping invalid file {result_file}: {e}")
        
        if not all_results:
            console.print("[red]✗[/red] No valid result files found")
            raise typer.Exit(1)
        
        # Generate report based on format
        if format.lower() == "html":
            generate_html_report(all_results, output, include_plots)
        elif format.lower() == "json":
            generate_json_report(all_results, output)
        else:
            console.print(f"[red]✗[/red] Unsupported format: {format}")
            raise typer.Exit(1)
        
        console.print(f"[green]✓[/green] Report generated: {output}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Report generation failed: {str(e)}")
        raise typer.Exit(1)

def generate_html_report(results: List[Dict], output_file: str, include_plots: bool = True):
    """Generate HTML report from experiment results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ETHPredict Experiment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 30px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ETHPredict Experiment Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Experiments: {len(results)}</p>
        </div>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <div class="metric">
                <strong>Average Sharpe Ratio:</strong> 
                {sum(r.get('metrics', {}).get('sharpe_ratio', 0) for r in results) / len(results):.4f}
            </div>
            <div class="metric">
                <strong>Best Sharpe Ratio:</strong> 
                {max(r.get('metrics', {}).get('sharpe_ratio', -999) for r in results):.4f}
            </div>
            <div class="metric">
                <strong>Success Rate:</strong> 
                {sum(1 for r in results if r.get('metrics', {}).get('sharpe_ratio', -999) > 0) / len(results) * 100:.1f}%
            </div>
        </div>
        
        <div class="section">
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Experiment ID</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Duration (s)</th>
                </tr>
    """
    
    for result in results:
        metrics = result.get('metrics', {})
        html_content += f"""
                <tr>
                    <td>{result.get('experiment_id', 'N/A')}</td>
                    <td>{metrics.get('sharpe_ratio', 'N/A'):.4f}</td>
                    <td>{metrics.get('max_drawdown', 'N/A'):.4f}</td>
                    <td>{result.get('duration', 'N/A'):.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def generate_json_report(results: List[Dict], output_file: str):
    """Generate JSON report from experiment results."""
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(results),
        "summary": {
            "avg_sharpe_ratio": sum(r.get('metrics', {}).get('sharpe_ratio', 0) for r in results) / len(results),
            "best_sharpe_ratio": max(r.get('metrics', {}).get('sharpe_ratio', -999) for r in results),
            "success_rate": sum(1 for r in results if r.get('metrics', {}).get('sharpe_ratio', -999) > 0) / len(results)
        },
        "experiments": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(report_data, f, indent=2)

@app.command()
def info():
    """Display system information and configuration."""
    console.print(Panel.fit("ℹ️  [bold blue]ETHPredict System Information[/bold blue]", border_style="blue"))
    
    # System info table
    info_table = Table(title="System Information", show_header=True, header_style="bold magenta")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Status", style="green")
    
    # Check PyTorch and CUDA
    import torch
    info_table.add_row("PyTorch Version", torch.__version__)
    info_table.add_row("CUDA Available", "✓" if torch.cuda.is_available() else "✗")
    if torch.cuda.is_available():
        info_table.add_row("CUDA Version", torch.version.cuda)
        info_table.add_row("GPU Count", str(torch.cuda.device_count()))
    
    # Check config files
    config_files = [
        ("Main Config", DEFAULT_CONFIG),
        ("Schema", "configs/schema.yaml")
    ]
    
    for config_type, config_path in config_files:
        exists = Path(config_path).exists()
        info_table.add_row(f"Config: {config_type}", "✓" if exists else "✗")
    
    console.print(info_table)
    
    # Display available configurations
    console.print("\n[bold]Available Configuration Files:[/bold]")
    console.print(f"  Default: [cyan]{DEFAULT_CONFIG}[/cyan]")
    
    # List all available configs
    configs_dir = Path("configs")
    if configs_dir.exists():
        console.print("  Available configs:")
        for cfg in sorted(configs_dir.glob("*.yml")):
            console.print(f"    - [cyan]{cfg}[/cyan]")

if __name__ == "__main__":
    app()
