"""Experiment parameter management and grid search."""

import itertools
from typing import Any, Dict, List, Optional
import yaml
import numpy as np
from scipy.stats import uniform, randint

class ExperimentManager:
    """Manages experiment parameters and grid search."""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self._validate_config()
        
    def _validate_config(self):
        """Validate experiment configuration."""
        required = ["experiment", "bars", "features", "model"]
        for key in required:
            if key not in self.base_config:
                raise ValueError(f"Missing required config section: {key}")
                
    def _expand_grid_params(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand grid parameters into list of configurations."""
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
        """Generate random parameter combinations."""
        trials = []
        for _ in range(n_trials):
            new_params = yaml.safe_load(yaml.dump(params))
            # Example: Random gamma in [0.1, 1.0]
            if "market_maker" in new_params and "gamma" in new_params["market_maker"]:
                new_params["market_maker"]["gamma"] = uniform(0.1, 0.9).rvs()
            trials.append(new_params)
        return trials
        
    def get_trials(self, mode: str = "grid", n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get list of trial configurations.
        
        Args:
            mode: One of "grid", "random", "bayesian"
            n_trials: Number of trials for random/bayesian modes
        """
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
        """Generate unique trial ID."""
        base_id = self.base_config["experiment"].get("id", "exp")
        return f"{base_id}_{trial_idx:04d}" 