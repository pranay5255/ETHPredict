"""Configuration loading for the active Lighter-only experiment runner."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema
import yaml


class BarType(str, Enum):
    TIME = "time"
    TICK = "tick"
    VOLUME = "volume"
    DOLLAR = "dollar"


class ModelType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


@dataclass
class Config:
    """Validated active experiment configuration."""

    experiment_id: str
    seed: int
    trials: int
    bar_type: BarType
    bar_threshold: float
    frac_diff_order: float
    features: List[str]
    model_type: ModelType
    model_params: Dict[str, Any]
    gamma: float
    hedge: bool
    inventory_limit: float
    quote_spread: float
    start_date: datetime
    end_date: datetime
    initial_capital: float
    time_interval: Optional[str] = None
    meta_labeling: bool = False


class ConfigManager:
    """Load and validate Lighter-only experiment configuration."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("configs")
        self._load_schema()

    def _load_schema(self):
        with open(self.config_dir / "schema.yaml", "r") as f:
            self.schema = yaml.safe_load(f)

    def load_config(self, config_paths: Union[str, List[str]]) -> Config:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        merged_config: Dict[str, Any] = {}
        for path in config_paths:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
                merged_config = self._deep_merge(merged_config, config)

        merged_config = self._process_template_vars(merged_config)
        try:
            jsonschema.validate(instance=merged_config, schema=self.schema)
        except jsonschema.exceptions.ValidationError as exc:
            raise ValueError(f"Schema validation error: {str(exc)}") from exc
        return self._dict_to_config(merged_config)

    def load_raw_config(self, config_paths: Union[str, List[str]]) -> Dict[str, Any]:
        if isinstance(config_paths, str):
            config_paths = [config_paths]
        merged_config: Dict[str, Any] = {}
        for path in config_paths:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
                merged_config = self._deep_merge(merged_config, config)
        return self._process_template_vars(merged_config)

    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _process_template_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config_str = yaml.dump(config)
        if "{{timestamp}}" in config_str:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_str = config_str.replace("{{timestamp}}", timestamp)
        for key, value in os.environ.items():
            config_str = config_str.replace(f"${{{key}}}", value)
        return yaml.safe_load(config_str)

    def _dict_to_config(self, config: Dict[str, Any]) -> Config:
        backtest = config.get("backtest", {})
        market_maker = config.get("market_maker", {})
        model = config.get("model", {})
        bars = config.get("bars", {})
        return Config(
            experiment_id=config["experiment"]["id"],
            seed=config["experiment"]["seed"],
            trials=config["experiment"]["trials"],
            bar_type=BarType(bars["type"]),
            bar_threshold=bars.get("threshold_usd", bars.get("threshold_volume", bars.get("threshold_ticks", 0))),
            time_interval=bars.get("time_interval"),
            frac_diff_order=config["features"]["frac_diff_order"],
            features=config["features"]["include"],
            model_type=ModelType(model["level0"]["algo"]),
            model_params=model["level0"].get("params", {}),
            meta_labeling=model.get("meta_labeling", False),
            gamma=market_maker.get("gamma", 0.5),
            hedge=market_maker.get("hedge", True),
            inventory_limit=market_maker.get("inventory_limit", 0),
            quote_spread=market_maker.get("quote_spread", 0.0),
            start_date=datetime.fromisoformat(str(backtest["start"])),
            end_date=datetime.fromisoformat(str(backtest["end"])),
            initial_capital=backtest["initial_capital"],
        )
