"""Main experiment orchestrator."""

import multiprocessing as mp
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm

from .gpu_manager import gpu_manager, init_gpu_manager
from .experiment import ExperimentManager
from ..adaptor import adapter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def _run_trial(cfg: Dict[str, Any], trial_id: str) -> Path:
    """Run a single trial with GPU resource management."""
    try:
        with gpu_manager.acquire():
            logger.info(f"Starting trial {trial_id}")
            data = adapter.load_data(cfg)
            X, y = adapter.make_features(cfg, data)
            res = adapter.train_models(cfg, X, y)
            
            # Save results
            path = RESULTS_DIR / f"{trial_id}.json"
            with open(path, "w") as f:
                json.dump({
                    "trial_id": trial_id,
                    "config": cfg,
                    "metrics": res
                }, f, indent=2)
            logger.info(f"Completed trial {trial_id}")
            return path
            
    except Exception as e:
        logger.error(f"Trial {trial_id} failed: {str(e)}")
        raise

def run_single(cfg: Dict[str, Any], tag: str = "") -> None:
    """Run a single experiment."""
    trial_id = tag or cfg.get("experiment", {}).get("id", "exp")
    _run_trial(cfg, trial_id)

def run_grid(
    cfg: Dict[str, Any],
    mode: str = "grid",
    max_par: int = 1,
    max_gpus: int = 1,
    n_trials: Optional[int] = None
) -> None:
    """Run multiple experiments in parallel.
    
    Args:
        cfg: Base configuration
        mode: One of "grid", "random", "bayesian"
        max_par: Maximum parallel processes
        max_gpus: Maximum GPUs to use
        n_trials: Number of trials for random/bayesian modes
    """
    # Initialize GPU manager
    init_gpu_manager(max_gpus)
    
    # Get trial configurations
    exp_manager = ExperimentManager(cfg)
    trials = exp_manager.get_trials(mode=mode, n_trials=n_trials)
    
    logger.info(f"Starting {len(trials)} trials in {mode} mode")
    
    # Run trials in parallel
    with mp.Pool(processes=max_par) as pool:
        trial_args = [
            (trial, exp_manager.get_trial_id(i, trial))
            for i, trial in enumerate(trials)
        ]
        
        # Use tqdm for progress bar
        for _ in tqdm(
            pool.starmap(_run_trial, trial_args),
            total=len(trials),
            desc="Running trials"
        ):
            pass

def generate_report(input_dir: Path, out_file: Path) -> None:
    """Generate HTML report from trial results."""
    lines = ["<html><body><h1>Experiment Results</h1>"]
    
    for p in sorted(Path(input_dir).glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        lines.append(f"<h2>Trial: {data['trial_id']}</h2>")
        lines.append("<h3>Configuration:</h3>")
        lines.append("<pre>")
        lines.append(json.dumps(data["config"], indent=2))
        lines.append("</pre>")
        lines.append("<h3>Metrics:</h3>")
        lines.append("<pre>")
        lines.append(json.dumps(data["metrics"], indent=2))
        lines.append("</pre>")
        lines.append("<hr>")
        
    lines.append("</body></html>")
    out_file.write_text("\n".join(lines))
