"""Configuration management for market making experiments."""

import os
import yaml
import jsonschema
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy.stats import uniform, randint

class ConfigManager:
    """Manages experiment configuration loading, validation and parameter sweeps."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize config manager.
        
        Args:
            schema_path: Path to JSON schema file. If None, uses default schema.
        """
        self.schema_path = schema_path or Path("configs/schema.yaml")
        self._load_schema()
        
    def _load_schema(self):
        """Load JSON schema for validation."""
        with open(self.schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)
            
    def _process_template_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process template variables in config (e.g. {{timestamp}})."""
        config_str = yaml.dump(config)
        
        # Replace timestamp
        if "{{timestamp}}" in config_str:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_str = config_str.replace("{{timestamp}}", timestamp)
            
        # Replace environment variables
        for key, value in os.environ.items():
            if f"${{{key}}}" in config_str:
                config_str = config_str.replace(f"${{{key}}}", value)
                
        return yaml.safe_load(config_str)
        
    def load_config(self, config_paths: Union[str, List[str]]) -> Dict[str, Any]:
        """Load and merge multiple config files.
        
        Args:
            config_paths: Path or list of paths to config files
            
        Returns:
            Merged configuration dictionary
        """
        if isinstance(config_paths, str):
            config_paths = [config_paths]
            
        merged_config = {}
        for path in config_paths:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                merged_config.update(config)
                
        # Process template variables
        merged_config = self._process_template_vars(merged_config)
        
        # Validate against schema
        jsonschema.validate(instance=merged_config, schema=self.schema)
        
        return merged_config
        
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
            
            # Randomize market maker gamma
            if "market_maker" in new_params and "gamma" in new_params["market_maker"]:
                new_params["market_maker"]["gamma"] = uniform(0.1, 0.9).rvs()
                
            # Randomize model hyperparameters
            if "model" in new_params and "level0" in new_params["model"]:
                model_params = new_params["model"]["level0"]["params"]
                if "max_depth" in model_params:
                    model_params["max_depth"] = randint(4, 10).rvs()
                if "eta" in model_params:
                    model_params["eta"] = uniform(0.01, 0.2).rvs()
                    
            trials.append(new_params)
        return trials
        
    def get_trials(self, config: Dict[str, Any], mode: str = "grid", n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get list of trial configurations.
        
        Args:
            config: Base configuration
            mode: One of "grid", "random", "bayesian"
            n_trials: Number of trials for random/bayesian modes
            
        Returns:
            List of trial configurations
        """
        if mode == "grid":
            return self._expand_grid_params(config)
        elif mode == "random":
            if not n_trials:
                raise ValueError("n_trials required for random mode")
            return self._generate_random_params(config, n_trials)
        elif mode == "bayesian":
            raise NotImplementedError("Bayesian optimization not yet implemented")
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    def get_trial_id(self, trial_idx: int, trial_params: Dict[str, Any]) -> str:
        """Generate unique trial ID."""
        base_id = trial_params["experiment"].get("id", "exp")
        return f"{base_id}_{trial_idx:04d}" 