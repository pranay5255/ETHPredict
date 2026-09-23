"""Trackio logging helpers for experiment runners."""

from __future__ import annotations

import math
import json
import shutil
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd


class TrackioLoggingError(RuntimeError):
    """A required research run was not recorded in Trackio."""


def trackio_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    tracking = config.get("tracking", {}) if isinstance(config, Mapping) else {}
    return dict((tracking or {}).get("trackio", {}) or {})


def trackio_enabled(config: Mapping[str, Any]) -> bool:
    return bool(trackio_settings(config).get("enabled", False))


def trackio_local_debug_override(config: Mapping[str, Any]) -> bool:
    settings = trackio_settings(config)
    return bool(settings.get("allow_local_debug_without_trackio", settings.get("local_debug", False)))


def enforce_trackio_policy(config: Mapping[str, Any], *, smoke: bool = False) -> None:
    if smoke or trackio_enabled(config) or trackio_local_debug_override(config):
        return
    raise RuntimeError(
        "Trackio is required for non-smoke research runs. "
        "Set tracking.trackio.enabled=true or tracking.trackio.allow_local_debug_without_trackio=true for explicit local debugging."
    )


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


def _write_receipt(path: Optional[Path], payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_local_run(project: str, name: str, status: str) -> None:
    """Check that Trackio's local reader can see the finished run."""
    executable = shutil.which("trackio")
    if executable is None:
        raise TrackioLoggingError("Trackio CLI is unavailable for local readback")
    result = subprocess.run(
        [executable, "get", "run", "--project", project, "--run", name, "--json"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    saved = json.loads(result.stdout)
    if saved.get("run") != name or f"trackio.status.{status}" not in saved.get("metrics", []):
        raise TrackioLoggingError(f"Trackio readback did not contain the finished run {name!r}")


def log_trackio_run(
    config: Mapping[str, Any],
    *,
    name: str,
    group: str,
    run_config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    status: str = "success",
    smoke: bool = False,
    receipt_path: Optional[Path] = None,
) -> bool:
    settings = trackio_settings(config)
    required = not smoke and not trackio_local_debug_override(config)
    project = str(settings.get("project", "ethpredict"))
    receipt = {"project": project, "run": name, "group": group, "status": status}
    if not settings.get("enabled", False):
        _write_receipt(receipt_path, {**receipt, "delivery": "disabled"})
        if required:
            raise TrackioLoggingError("Trackio is required for non-smoke research runs")
        return False

    initialized = False
    try:
        import trackio
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

        trackio.init(**init_kwargs)
        initialized = True
        flat_metrics = flatten_numeric(dict(metrics or {}))
        flat_metrics[f"trackio.status.{status}"] = 1.0
        if flat_metrics:
            trackio.log(flat_metrics)
        trackio.finish()
        initialized = False
        if settings.get("space_id") or settings.get("server_url"):
            verification = "remote_unverified"
        elif required:
            _verify_local_run(project, name, status)
            verification = "local_readback"
        else:
            verification = "not_required"
        _write_receipt(receipt_path, {**receipt, "delivery": "logged", "verification": verification})
        return True
    except Exception as exc:
        if initialized:
            try:
                trackio.finish()
            except Exception:
                pass
        _write_receipt(receipt_path, {**receipt, "delivery": "failed", "error": str(exc)})
        if required:
            raise TrackioLoggingError(f"Trackio logging failed for run {name!r}: {exc}") from exc
        warnings.warn(f"Trackio logging failed for run {name!r}: {exc}", RuntimeWarning)
        return False
