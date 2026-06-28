"""Configuration management for market making experiments."""

import os
import yaml
import jsonschema
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy.stats import uniform, randint, loguniform

# Configuration Types
class BarType(str, Enum):
    TIME = "time"
    TICK = "tick"
    VOLUME = "volume"
    DOLLAR = "dollar"

class ModelType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"

class BribeMode(str, Enum):
    PERCENTILE = "percentile"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"

class SimMode(str, Enum):
    AMM = "amm"
    LOB = "lob"

@dataclass
class Config:
    """Complete experiment configuration."""
    # Experiment settings
    experiment_id: str
    seed: int
    trials: int
    
    # Bar construction
    bar_type: BarType
    bar_threshold: float
    time_interval: Optional[str] = None
    
    # Feature engineering
    frac_diff_order: float
    features: List[str]
    
    # Model settings
    model_type: ModelType
    model_params: Dict[str, Any]
    meta_labeling: bool = False
    
    # Market making
    gamma: float
    hedge: bool
    inventory_limit: float
    quote_spread: float
    
    # Bribe settings
    bribe_mode: BribeMode
    bribe_value: float
    
    # Backtest settings
    start_date: str
    end_date: str
    initial_capital: float
    
    # Simulation settings
    sim_mode: SimMode
    sim_seed: int
    sim_params: Dict[str, Any]

class ConfigManager:
    """Manages experiment configuration loading, validation and parameter sweeps."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize config manager.
        
        Args:
            config_dir: Directory containing config files. If None, uses configs/
        """
        self.config_dir = config_dir or Path("configs")
        self._load_schema()
        
    def _load_schema(self):
        """Load JSON schema for validation."""
        with open(self.config_dir / "schema.yaml", 'r') as f:
            self.schema = yaml.safe_load(f)
            
    def load_config(self, config_paths: Union[str, List[str]]) -> Config:
        """Load and validate configuration.
        
        Args:
            config_paths: Path or list of paths to config files
            
        Returns:
            Validated configuration object
            
        Raises:
            ValueError: If configuration is invalid
        """
        if isinstance(config_paths, str):
            config_paths = [config_paths]
            
        # Load and merge configs
        merged_config = {}
        for path in config_paths:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                merged_config = self._deep_merge(merged_config, config)
                
        # Process template variables
        merged_config = self._process_template_vars(merged_config)
        
        # Validate against schema
        try:
            jsonschema.validate(instance=merged_config, schema=self.schema)
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Schema validation error: {str(e)}")
            
        # Convert to Config object
        return self._dict_to_config(merged_config)
        
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
        
    def _process_template_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process template variables in config."""
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
        
    def _dict_to_config(self, config: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        return Config(
            # Experiment settings
            experiment_id=config["experiment"]["id"],
            seed=config["experiment"]["seed"],
            trials=config["experiment"]["trials"],
            
            # Bar construction
            bar_type=BarType(config["bars"]["type"]),
            bar_threshold=config["bars"]["threshold_usd"],
            time_interval=config["bars"].get("time_interval"),
            
            # Feature engineering
            frac_diff_order=config["features"]["frac_diff_order"],
            features=config["features"]["include"],
            
            # Model settings
            model_type=ModelType(config["model"]["level0"]["algo"]),
            model_params=config["model"]["level0"]["params"],
            meta_labeling=config["model"]["meta_labeling"],
            
            # Market making
            gamma=config["market_maker"]["gamma"],
            hedge=config["market_maker"]["hedge"],
            inventory_limit=config["market_maker"]["inventory_limit"],
            quote_spread=config["market_maker"]["quote_spread"],
            
            # Bribe settings
            bribe_mode=BribeMode(config["bribe"]["mode"]),
            bribe_value=config["bribe"].get("percentile") or config["bribe"].get("fixed_amount", 0.0),
            
            # Backtest settings
            start_date=datetime.fromisoformat(config["backtest"]["start"]),
            end_date=datetime.fromisoformat(config["backtest"]["end"]),
            initial_capital=config["backtest"]["initial_capital"],
            
            # Simulation settings
            sim_mode=SimMode(config["sim"]["mode"]),
            sim_seed=config["sim"]["seed"],
            sim_params=config["sim"].get("amm") or config["sim"].get("lob", {})
        )
        
    def generate_trials(self, config: Config, mode: str = "grid", n_trials: Optional[int] = None) -> List[Config]:
        """Generate parameter sweep trials.
        
        Args:
            config: Base configuration
            mode: One of "grid", "random"
            n_trials: Number of trials for random mode
            
        Returns:
            List of trial configurations
        """
        if mode == "grid":
            return self._generate_grid(config)
        elif mode == "random":
            if not n_trials:
                raise ValueError("n_trials required for random mode")
            return self._generate_random(config, n_trials)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    def _generate_grid(self, config: Config) -> List[Config]:
        """Generate grid search trials."""
        # TODO: Implement grid search
        return [config]
        
    def _generate_random(self, config: Config, n_trials: int) -> List[Config]:
        """Generate random trials."""
        trials = []
        for _ in range(n_trials):
            new_config = Config(
                experiment_id=config.experiment_id,
                seed=np.random.randint(0, 2**32),
                trials=config.trials,
                
                bar_type=config.bar_type,
                bar_threshold=loguniform(10000, 1000000).rvs(),
                time_interval=config.time_interval,
                
                frac_diff_order=uniform(0.1, 0.9).rvs(),
                features=config.features,
                
                model_type=config.model_type,
                model_params={
                    "max_depth": randint(4, 10).rvs(),
                    "eta": uniform(0.01, 0.2).rvs(),
                    **config.model_params
                },
                meta_labeling=config.meta_labeling,
                
                gamma=uniform(0.1, 0.9).rvs(),
                hedge=config.hedge,
                inventory_limit=loguniform(1000, 100000).rvs(),
                quote_spread=loguniform(0.0001, 0.01).rvs(),
                
                bribe_mode=config.bribe_mode,
                bribe_value=uniform(0.5, 0.99).rvs(),
                
                start_date=config.start_date,
                end_date=config.end_date,
                initial_capital=config.initial_capital,
                
                sim_mode=config.sim_mode,
                sim_seed=np.random.randint(0, 2**32),
                sim_params=config.sim_params
            )
            trials.append(new_config)
        return trials 