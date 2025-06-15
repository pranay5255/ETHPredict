import os
import json
import time
import logging
import threading
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
import optuna
from tqdm import tqdm
from joblib import Parallel, delayed
import torch

@dataclass
class TrialResult:
    """Container for trial results and metadata."""
    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    status: str  # "success", "failed", "in_progress"
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    gpu_id: Optional[int] = None

class ExperimentOrchestrator:
    def __init__(
        self,
        search_space: Dict[str, Dict[str, Any]],
        search_type: str = "grid",
        n_trials: int = 1000,
        n_workers: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
        checkpoint_dir: str = "experiment_checkpoints",
        metric_name: str = "loss",
        maximize: bool = False
    ):
        """
        Initialize the experiment orchestrator.
        
        Args:
            search_space: Dictionary defining the parameter search space
            search_type: One of "grid", "random", or "bayesian"
            n_trials: Number of trials to run
            n_workers: Number of parallel workers (defaults to CPU count)
            gpu_ids: List of GPU IDs to use (None for CPU-only)
            checkpoint_dir: Directory to save checkpoints
            metric_name: Name of the metric to optimize
            maximize: Whether to maximize the metric (True) or minimize it (False)
        """
        self.search_space = search_space
        self.search_type = search_type.lower()
        self.n_trials = n_trials
        self.n_workers = n_workers or multiprocessing.cpu_count()
        self.gpu_ids = gpu_ids or []
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric_name = metric_name
        self.maximize = maximize
        
        # Initialize state
        self.trial_results: Dict[int, TrialResult] = {}
        self.current_trial_id = 0
        self.gpu_semaphore = threading.Semaphore(len(self.gpu_ids)) if self.gpu_ids else None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.checkpoint_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load existing checkpoint if available
        self._load_checkpoint()

    def _generate_grid_search_params(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        param_values = []
        for param_name, param_config in self.search_space.items():
            if param_config["type"] == "continuous":
                values = np.linspace(
                    param_config["range"][0],
                    param_config["range"][1],
                    param_config.get("n_points", 10)
                )
            elif param_config["type"] == "discrete":
                values = param_config["values"]
            elif param_config["type"] == "categorical":
                values = param_config["values"]
            else:
                raise ValueError(f"Unknown parameter type: {param_config['type']}")
            param_values.append((param_name, values))
        
        # Generate all combinations
        from itertools import product
        param_names = [p[0] for p in param_values]
        param_combinations = list(product(*[p[1] for p in param_values]))
        return [dict(zip(param_names, combo)) for combo in param_combinations]

    def _generate_random_search_params(self) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""
        params_list = []
        for _ in range(self.n_trials):
            params = {}
            for param_name, param_config in self.search_space.items():
                if param_config["type"] == "continuous":
                    params[param_name] = np.random.uniform(
                        param_config["range"][0],
                        param_config["range"][1]
                    )
                elif param_config["type"] == "discrete":
                    params[param_name] = np.random.choice(param_config["values"])
                elif param_config["type"] == "categorical":
                    params[param_name] = np.random.choice(param_config["values"])
            params_list.append(params)
        return params_list

    def _bayesian_search_objective(self, trial: optuna.Trial) -> float:
        """Objective function for Bayesian optimization."""
        params = {}
        for param_name, param_config in self.search_space.items():
            if param_config["type"] == "continuous":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["range"][0],
                    param_config["range"][1]
                )
            elif param_config["type"] == "discrete":
                params[param_name] = trial.suggest_int(
                    param_name,
                    min(param_config["values"]),
                    max(param_config["values"])
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["values"]
                )
        
        # Run the experiment
        result = self._run_single_trial(params)
        return result.metrics[self.metric_name]

    def _allocate_gpu(self) -> Optional[int]:
        """Allocate a GPU if available."""
        if not self.gpu_ids or not self.gpu_semaphore:
            return None
        
        self.gpu_semaphore.acquire()
        available_gpu = next(
            (gpu_id for gpu_id in self.gpu_ids 
             if not torch.cuda.is_initialized() or 
             torch.cuda.memory_allocated(gpu_id) == 0),
            self.gpu_ids[0]
        )
        return available_gpu

    def _release_gpu(self, gpu_id: Optional[int]):
        """Release a GPU."""
        if gpu_id is not None and self.gpu_semaphore:
            self.gpu_semaphore.release()

    def _run_single_trial(self, params: Dict[str, Any]) -> TrialResult:
        """Run a single trial with the given parameters."""
        trial_id = self.current_trial_id
        self.current_trial_id += 1
        
        result = TrialResult(
            trial_id=trial_id,
            params=params,
            metrics={},
            status="in_progress",
            start_time=time.time()
        )
        
        try:
            # Allocate GPU if available
            gpu_id = self._allocate_gpu()
            result.gpu_id = gpu_id
            
            # Run the experiment
            metrics = self.experiment_fn(params, gpu_id)
            result.metrics = metrics
            result.status = "success"
            
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            self.logger.error(f"Trial {trial_id} failed: {str(e)}")
            
        finally:
            result.end_time = time.time()
            self._release_gpu(gpu_id)
            self.trial_results[trial_id] = result
            self._save_checkpoint()
            
        return result

    def _save_checkpoint(self):
        """Save the current state of the experiment."""
        checkpoint = {
            "trial_results": {str(k): asdict(v) for k, v in self.trial_results.items()},
            "current_trial_id": self.current_trial_id,
            "search_space": self.search_space,
            "search_type": self.search_type,
            "n_trials": self.n_trials
        }
        
        with open(self.checkpoint_dir / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self):
        """Load the experiment state from checkpoint."""
        checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            
            self.trial_results = {
                int(k): TrialResult(**v) 
                for k, v in checkpoint["trial_results"].items()
            }
            self.current_trial_id = checkpoint["current_trial_id"]
            self.logger.info(f"Loaded checkpoint with {len(self.trial_results)} completed trials")

    def run_experiment(self, experiment_fn: Callable[[Dict[str, Any], Optional[int]], Dict[str, float]]):
        """
        Run the experiment with the given experiment function.
        
        Args:
            experiment_fn: Function that takes parameters and optional GPU ID,
                          returns a dictionary of metrics
        """
        self.experiment_fn = experiment_fn
        
        if self.search_type == "grid":
            param_combinations = self._generate_grid_search_params()
            self.n_trials = len(param_combinations)
        elif self.search_type == "random":
            param_combinations = self._generate_random_search_params()
        elif self.search_type == "bayesian":
            study = optuna.create_study(
                direction="maximize" if self.maximize else "minimize"
            )
            study.optimize(
                self._bayesian_search_objective,
                n_trials=self.n_trials,
                n_jobs=self.n_workers
            )
            return study
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")
        
        # Run trials in parallel
        with tqdm(total=self.n_trials, desc="Running trials") as pbar:
            results = Parallel(n_jobs=self.n_workers)(
                delayed(self._run_single_trial)(params)
                for params in param_combinations
            )
            pbar.update(len(results))
        
        return self.trial_results

    def get_best_trial(self) -> Optional[TrialResult]:
        """Get the best trial based on the metric."""
        if not self.trial_results:
            return None
            
        successful_trials = [
            t for t in self.trial_results.values()
            if t.status == "success"
        ]
        
        if not successful_trials:
            return None
            
        return max(
            successful_trials,
            key=lambda x: x.metrics[self.metric_name] if self.maximize 
            else -x.metrics[self.metric_name]
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment results."""
        successful_trials = [t for t in self.trial_results.values() if t.status == "success"]
        failed_trials = [t for t in self.trial_results.values() if t.status == "failed"]
        
        return {
            "total_trials": len(self.trial_results),
            "successful_trials": len(successful_trials),
            "failed_trials": len(failed_trials),
            "best_trial": asdict(self.get_best_trial()) if self.get_best_trial() else None,
            "average_metrics": {
                metric: np.mean([t.metrics[metric] for t in successful_trials])
                for metric in successful_trials[0].metrics.keys()
            } if successful_trials else {}
        } 