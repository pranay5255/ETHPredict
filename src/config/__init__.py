"""Configuration management for market making experiments."""

from pathlib import Path
from typing import List, Optional, Union

from .core.loader import ConfigLoader
from .core.types import Config
from .validators.schema import SchemaValidator
from .sweeps.generator import ParameterSweep

class ConfigManager:
    """Main configuration management interface."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            base_dir: Base directory for config files. If None, uses configs/
        """
        self.base_dir = base_dir or Path("configs")
        self.loader = ConfigLoader(self.base_dir)
        self.validator = SchemaValidator(self.base_dir / "schema.yaml")
        self.sweep = ParameterSweep()
        
    def load_config(self, config_paths: Union[str, List[str]]) -> Config:
        """Load and validate configuration.
        
        Args:
            config_paths: Path or list of paths to config files
            
        Returns:
            Validated configuration object
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Load configuration
        config = self.loader.load_config(config_paths)
        
        # Validate configuration
        errors = self.validator.validate_config(config)
        if errors:
            raise ValueError(f"Invalid configuration:\n" + "\n".join(errors))
            
        return config
        
    def load_experiment(self, experiment_name: str) -> Config:
        """Load and validate experiment configuration.
        
        Args:
            experiment_name: Name of experiment config to load
            
        Returns:
            Validated experiment configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Load configuration
        config = self.loader.load_experiment_config(experiment_name)
        
        # Validate configuration
        errors = self.validator.validate_config(config)
        if errors:
            raise ValueError(f"Invalid configuration:\n" + "\n".join(errors))
            
        return config
        
    def generate_trials(self, config: Config, mode: str = "grid", n_trials: Optional[int] = None) -> List[Config]:
        """Generate parameter sweep trials.
        
        Args:
            config: Base configuration
            mode: One of "grid", "random", "bayesian"
            n_trials: Number of trials for random/bayesian modes
            
        Returns:
            List of trial configurations
            
        Raises:
            ValueError: If mode is invalid or n_trials is missing
        """
        if mode == "grid":
            return self.sweep.generate_grid(config)
        elif mode == "random":
            if not n_trials:
                raise ValueError("n_trials required for random mode")
            return self.sweep.generate_random(config, n_trials)
        elif mode == "bayesian":
            raise NotImplementedError("Bayesian optimization not yet implemented")
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    def validate_config(self, config: Config) -> List[str]:
        """Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages, empty if valid
        """
        return self.validator.validate_config(config) 