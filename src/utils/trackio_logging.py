"""Optional Trackio logging helpers for experiment runners."""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd


def trackio_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    tracking = config.get("tracking", {}) if isinstance(config, Mapping) else {}
    return dict((tracking or {}).get("trackio", {}) or {})


def trackio_enabled(config: Mapping[str, Any]) -> bool:
    return bool(trackio_settings(config).get("enabled", False))


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def flatten_numeric(payload: Mapping[str, Any], *, prefix: str = "", sep: str = ".") -> Dict[str, float]:
    flat: Dict[str, float] = {}

    def visit(key: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                child_name = f"{key}{sep}{child_key}" if key else str(child_key)
                visit(child_name, child_value)
            return
        if isinstance(value, (list, tuple)):
            return
        if isinstance(value, (np.integer,)):
            flat[key] = int(value)
            return
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, bool):
            flat[key] = float(value)
            return
        if isinstance(value, int):
            flat[key] = value
            return
        if isinstance(value, float) and math.isfinite(value):
            flat[key] = value

    visit(prefix, payload)
    return {key: value for key, value in flat.items() if key}


def log_trackio_run(
    config: Mapping[str, Any],
    *,
    name: str,
    group: str,
    run_config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    status: str = "success",
) -> bool:
    settings = trackio_settings(config)
    if not settings.get("enabled", False):
        return False

    try:
        import trackio
    except Exception as exc:  # pragma: no cover - only hit when optional dep missing
        warnings.warn(f"Trackio logging skipped because import failed: {exc}", RuntimeWarning)
        return False

    project = str(settings.get("project", "ethpredict"))
    init_kwargs: Dict[str, Any] = {
        "project": project,
        "name": name,
        "group": group,
        "config": json_ready(
            {
                **dict(run_config or {}),
                "artifact_paths": dict(artifacts or {}),
                "trackio_status": status,
            }
        ),
        "auto_log_gpu": bool(settings.get("auto_log_gpu", False)),
        "auto_log_cpu": bool(settings.get("auto_log_cpu", False)),
    }
    for key in ["space_id", "server_url", "dataset_id", "bucket_id", "resume", "webhook_url", "webhook_min_level"]:
        if settings.get(key) is not None:
            init_kwargs[key] = settings[key]
    for key in ["gpu_log_interval", "cpu_log_interval"]:
        if settings.get(key) is not None:
            init_kwargs[key] = float(settings[key])

    initialized = False
    try:
        trackio.init(**init_kwargs)
        initialized = True
        flat_metrics = flatten_numeric(dict(metrics or {}))
        flat_metrics[f"trackio.status.{status}"] = 1.0
        if flat_metrics:
            trackio.log(flat_metrics)
        trackio.finish()
        return True
    except Exception as exc:  # pragma: no cover - defensive integration guard
        warnings.warn(f"Trackio logging failed for run {name!r}: {exc}", RuntimeWarning)
        if initialized:
            try:
                trackio.finish()
            except Exception:
                pass
        return False
