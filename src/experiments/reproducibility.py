"""Deterministic comparison helpers for AFML run manifests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def _load_manifest(path: Path) -> Dict[str, Any]:
    candidate = path / "manifest.json" if path.is_dir() else path
    with candidate.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get(payload: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _raw_hashes(manifest: Mapping[str, Any]) -> Dict[str, str]:
    files = _get(manifest, "raw_data.files", []) or _get(manifest, "run_identity.raw_data.files", []) or []
    return {str(item.get("path")): str(item.get("sha256")) for item in files if item.get("path")}


def _trial_map(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    trials = manifest.get("trials", []) or []
    return {str(trial.get("trial_id")): trial for trial in trials if trial.get("trial_id") is not None}


def _metric_variance(left: Any, right: Any, *, tolerance: float, prefix: str = "") -> Dict[str, Any]:
    diffs: Dict[str, Any] = {}
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            child = _metric_variance(left.get(key), right.get(key), tolerance=tolerance, prefix=f"{prefix}.{key}" if prefix else str(key))
            diffs.update(child)
        return diffs
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        a = float(left)
        b = float(right)
        delta = abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf
        if delta > tolerance:
            diffs[prefix] = {"left": a, "right": b, "absolute_delta": delta, "tolerance": tolerance}
    elif left != right and prefix:
        diffs[prefix] = {"left": left, "right": right}
    return diffs


def compare_run_manifests(left: Path, right: Path, *, metric_tolerance: float = 1e-9) -> Dict[str, Any]:
    """Compare two v2 run manifests or run directories for reproducibility drift."""

    left_manifest = _load_manifest(left)
    right_manifest = _load_manifest(right)
    left_trials = _trial_map(left_manifest)
    right_trials = _trial_map(right_manifest)
    trial_ids = sorted(set(left_trials) | set(right_trials))

    split_diffs = {
        trial_id: {
            "left": _get(left_trials.get(trial_id, {}), "split_manifest_hash"),
            "right": _get(right_trials.get(trial_id, {}), "split_manifest_hash"),
        }
        for trial_id in trial_ids
        if _get(left_trials.get(trial_id, {}), "split_manifest_hash") != _get(right_trials.get(trial_id, {}), "split_manifest_hash")
    }
    metric_diffs = {
        trial_id: _metric_variance(
            _get(left_trials.get(trial_id, {}), "metrics", {}),
            _get(right_trials.get(trial_id, {}), "metrics", {}),
            tolerance=metric_tolerance,
        )
        for trial_id in trial_ids
    }
    metric_diffs = {trial_id: diffs for trial_id, diffs in metric_diffs.items() if diffs}

    checks = {
        "config_hash_match": _get(left_manifest, "run_identity.resolved_config_hash") == _get(right_manifest, "run_identity.resolved_config_hash"),
        "raw_hashes_match": _raw_hashes(left_manifest) == _raw_hashes(right_manifest),
        "trial_count_match": int(left_manifest.get("trial_count", len(left_trials)) or 0) == int(right_manifest.get("trial_count", len(right_trials)) or 0),
        "trial_ids_match": sorted(left_trials) == sorted(right_trials),
        "split_hashes_match": not split_diffs,
        "metrics_within_tolerance": not metric_diffs,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "left": str(left),
        "right": str(right),
        "raw_hashes": {"left": _raw_hashes(left_manifest), "right": _raw_hashes(right_manifest)},
        "split_diffs": split_diffs,
        "metric_diffs": metric_diffs,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two AFML v2 run manifests for reproducibility drift.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)
    report = compare_run_manifests(args.left, args.right, metric_tolerance=args.metric_tolerance)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
