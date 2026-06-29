"""Configuration loading for the active Lighter-only experiment runner."""

from __future__ import annotations

import itertools
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

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
    LSTM = "lstm"


@dataclass
class Config:
    """Validated active experiment configuration.

    The scalar fields preserve the legacy public interface. The optional mapping
    fields expose the v2 research config without forcing callers through the
    older market-making oriented shape.
    """

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
    config_version: int = 1
    raw_config: Optional[Dict[str, Any]] = None
    targets: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    labels: Optional[Dict[str, Any]] = None
    training: Optional[Dict[str, Any]] = None
    costs: Optional[Dict[str, Any]] = None
    alpha_backtest: Optional[Dict[str, Any]] = None
    search: Optional[Dict[str, Any]] = None
    tracking: Optional[Dict[str, Any]] = None


class ConfigManager:
    """Load and validate Lighter-only experiment configuration."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("configs")
        self._load_schema()

    def _load_schema(self):
        with open(self.config_dir / "schema.yaml", "r", encoding="utf-8") as f:
            self.schema = yaml.safe_load(f)

    def load_config(self, config_paths: Union[str, List[str]]) -> Config:
        merged_config = self.load_raw_config(config_paths)
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
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
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

    def _is_v2(self, config: Mapping[str, Any]) -> bool:
        targets = config.get("targets")
        return config.get("version") == 2 or (isinstance(targets, Mapping) and "horizons" in targets)

    def _dict_to_config(self, config: Dict[str, Any]) -> Config:
        if self._is_v2(config):
            return self._dict_to_v2_config(config)
        return self._dict_to_legacy_config(config)

    def _dict_to_legacy_config(self, config: Dict[str, Any]) -> Config:
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
            frac_diff_order=config.get("features", {}).get("frac_diff_order", 0.0),
            features=list(config.get("features", {}).get("include", [])),
            model_type=ModelType(model.get("level0", {}).get("algo", "xgboost")),
            model_params=model.get("level0", {}).get("params", {}),
            meta_labeling=model.get("meta_labeling", False),
            gamma=market_maker.get("gamma", 0.5),
            hedge=market_maker.get("hedge", True),
            inventory_limit=market_maker.get("inventory_limit", 0),
            quote_spread=market_maker.get("quote_spread", 0.0),
            start_date=datetime.fromisoformat(str(backtest.get("start", "1970-01-01"))),
            end_date=datetime.fromisoformat(str(backtest.get("end", "1970-01-02"))),
            initial_capital=backtest.get("initial_capital", 0.0),
            raw_config=config,
        )

    def _dict_to_v2_config(self, config: Dict[str, Any]) -> Config:
        data = config.get("data", {})
        lighter = data.get("lighter", {})
        bars = config.get("bars", {}) or {"type": "time"}
        features = config.get("features", {})
        model_base = config.get("model", {}).get("base", {})
        costs = config.get("costs", {})
        alpha = config.get("alpha_backtest", {})
        return Config(
            experiment_id=config["experiment"]["id"],
            seed=int(config["experiment"].get("seed", 42)),
            trials=int(config["experiment"].get("trials", config.get("search", {}).get("max_trials", 1))),
            bar_type=BarType(bars.get("type", "time")),
            bar_threshold=float(bars.get("threshold_usd", bars.get("threshold_volume", bars.get("threshold_ticks", 0.0)))),
            time_interval=bars.get("time_interval", data.get("granularity")),
            frac_diff_order=float(features.get("frac_diff_order", 0.0)),
            features=list(features.get("include", [])),
            model_type=ModelType(model_base.get("type", "lstm")),
            model_params=dict(model_base),
            meta_labeling=bool(config.get("model", {}).get("meta_labeler", {}).get("enabled", True)),
            gamma=0.0,
            hedge=False,
            inventory_limit=float(alpha.get("max_position_notional", 0.0)),
            quote_spread=float(costs.get("spread_bps", 0.0)) / 10_000.0,
            start_date=datetime.fromisoformat(str(lighter.get("start_date", "1970-01-01"))),
            end_date=datetime.fromisoformat(str(lighter.get("end_date", "1970-01-02"))),
            initial_capital=float(alpha.get("initial_capital", 0.0)),
            config_version=2,
            raw_config=config,
            targets=config.get("targets"),
            validation=config.get("validation"),
            labels=config.get("labels"),
            training=config.get("training"),
            costs=config.get("costs"),
            alpha_backtest=config.get("alpha_backtest"),
            search=config.get("search"),
            tracking=config.get("tracking"),
        )


def get_dotted(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def set_dotted(config: Dict[str, Any], path: str, value: Any) -> None:
    current: Dict[str, Any] = config
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def apply_dotted_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    updated = deepcopy(dict(config))
    for path, value in overrides.items():
        set_dotted(updated, str(path), value)
    return updated


def iter_grid_overrides(search: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if search.get("mode", "grid") != "grid":
        raise ValueError(f"Unsupported search mode: {search.get('mode')}")
    spaces = search.get("spaces", {}) or {}
    items: List[Tuple[str, List[Any]]] = []
    for params in spaces.values():
        if not isinstance(params, Mapping):
            continue
        for path, raw_values in params.items():
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            items.append((str(path), list(values)))
    if not items:
        return [{}]
    paths = [item[0] for item in items]
    values = [item[1] for item in items]
    overrides = [dict(zip(paths, combo)) for combo in itertools.product(*values)]
    max_trials = search.get("max_trials")
    if max_trials is not None:
        overrides = overrides[: int(max_trials)]
    return overrides


def expand_grid_search(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand ``search.spaces`` into trial configs using dotted override paths."""

    search = config.get("search", {}) if isinstance(config, Mapping) else {}
    trials: List[Dict[str, Any]] = []
    for idx, overrides in enumerate(iter_grid_overrides(search)):
        trials.append(
            {
                "trial_index": idx,
                "trial_id": f"grid_{idx:03d}",
                "overrides": overrides,
                "config": apply_dotted_overrides(config, overrides),
            }
        )
    return trials
