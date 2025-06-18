"""Main experiment orchestrator (consolidated)."""

import multiprocessing as mp
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, List, Union
from tqdm import tqdm
import os
import yaml
import jsonschema
from datetime import datetime
import numpy as np
from scipy.stats import uniform, randint
import threading
from contextlib import contextmanager

from ..adaptor import adapter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# --- GPU Manager (from gpu_manager.py) ---
class GPUSemaphore:
    """Manages GPU access across multiple processes."""
    def __init__(self, max_gpus: int = 1):
        self._semaphore = threading.Semaphore(max_gpus)
        self._lock = threading.Lock()
        self._active_gpus = 0
    @contextmanager
    def acquire(self):
        self._semaphore.acquire()
        with self._lock:
            self._active_gpus += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_gpus -= 1
            self._semaphore.release()
    @property
    def active_gpus(self) -> int:
        with self._lock:
            return self._active_gpus

gpu_manager: Optional[GPUSemaphore] = None

def init_gpu_manager(max_gpus: int = 1):
    global gpu_manager
    gpu_manager = GPUSemaphore(max_gpus)

# --- ExperimentManager (from experiment.py, merged with config.py logic) ---
class ExperimentManager:
    """Manages experiment parameters and grid/random search."""
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self._validate_config()
    def _validate_config(self):
        required = ["experiment", "bars", "features", "model"]
        for key in required:
            if key not in self.base_config:
                raise ValueError(f"Missing required config section: {key}")
    def _expand_grid_params(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        grid_params = {k: v for k, v in params.items() if isinstance(v, list)}
        if not grid_params:
            return [params]
        keys, values = zip(*grid_params.items())
        trials = []
        for combo in itertools.product(*values):
            new_params = yaml.safe_load(yaml.dump(params))
            for key, val in zip(keys, combo):
                new_params[key] = val
            trials.append(new_params)
        return trials
    def _generate_random_params(self, params: Dict[str, Any], n_trials: int) -> List[Dict[str, Any]]:
        trials = []
        for _ in range(n_trials):
            new_params = yaml.safe_load(yaml.dump(params))
            if "market_maker" in new_params and "gamma" in new_params["market_maker"]:
                new_params["market_maker"]["gamma"] = uniform(0.1, 0.9).rvs()
            if "model" in new_params and "level0" in new_params["model"]:
                model_params = new_params["model"]["level0"]["params"]
                if "max_depth" in model_params:
                    model_params["max_depth"] = randint(4, 10).rvs()
                if "eta" in model_params:
                    model_params["eta"] = uniform(0.01, 0.2).rvs()
            trials.append(new_params)
        return trials
    def get_trials(self, mode: str = "grid", n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        if mode == "grid":
            return self._expand_grid_params(self.base_config)
        elif mode == "random":
            if not n_trials:
                raise ValueError("n_trials required for random mode")
            return self._generate_random_params(self.base_config, n_trials)
        elif mode == "bayesian":
            raise NotImplementedError("Bayesian optimization not yet implemented")
        else:
            raise ValueError(f"Unknown mode: {mode}")
    def get_trial_id(self, trial_idx: int, trial_params: Dict[str, Any]) -> str:
        base_id = self.base_config["experiment"].get("id", "exp")
        return f"{base_id}_{trial_idx:04d}"

# --- ConfigManager (from config.py, for config loading/validation) ---
class ConfigManager:
    """Manages experiment configuration loading, validation and parameter sweeps."""
    def __init__(self, schema_path: Optional[Path] = None):
        self.schema_path = schema_path or Path("configs/schema.yaml")
        self._load_schema()
    def _load_schema(self):
        with open(self.schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)
    def _process_template_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config_str = yaml.dump(config)
        if "{{timestamp}}" in config_str:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_str = config_str.replace("{{timestamp}}", timestamp)
        for key, value in os.environ.items():
            if f"${{{key}}}" in config_str:
                config_str = config_str.replace(f"${{{key}}}", value)
        return yaml.safe_load(config_str)
    def load_config(self, config_paths: Union[str, List[str]]) -> Dict[str, Any]:
        if isinstance(config_paths, str):
            config_paths = [config_paths]
        merged_config = {}
        for path in config_paths:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                merged_config.update(config)
        merged_config = self._process_template_vars(merged_config)
        jsonschema.validate(instance=merged_config, schema=self.schema)
        return merged_config

# --- Orchestration logic (from main.py) ---
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
