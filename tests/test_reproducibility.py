import json
from pathlib import Path

import pytest

from src.experiments.meta_labeling_mvp import _record_final_test_evaluation
from src.experiments.reproducibility import compare_run_manifests


def _write_manifest(run_dir: Path, *, raw_hash: str = "abc", split_hash: str = "split", net_pnl: float = 1.0):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_identity": {"resolved_config_hash": "cfg"},
        "raw_data": {"files": [{"path": "data/raw/file.csv", "sha256": raw_hash}]},
        "trial_count": 1,
        "trials": [
            {
                "trial_id": "grid_000",
                "split_manifest_hash": split_hash,
                "metrics": {"validation": {"net_pnl": net_pnl}, "test": {"net_pnl": net_pnl}},
            }
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_compare_run_manifests_detects_raw_split_and_metric_drift(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_manifest(left)
    _write_manifest(right)

    assert compare_run_manifests(left, right)["status"] == "pass"

    _write_manifest(right, raw_hash="changed", split_hash="changed", net_pnl=2.0)
    report = compare_run_manifests(left, right, metric_tolerance=0.01)

    assert report["status"] == "fail"
    assert report["checks"]["raw_hashes_match"] is False
    assert report["checks"]["split_hashes_match"] is False
    assert report["checks"]["metrics_within_tolerance"] is False


def test_final_test_reuse_guard_counts_and_blocks_non_smoke_reuse(tmp_path):
    config = {"research": {"final_test_guard": {"do_not_reuse_test": True}}}

    first = _record_final_test_evaluation(config, tmp_path / "run_a", trial_id="grid_000", split_hash="split", smoke=False)
    assert first["count"] == 1

    with pytest.raises(RuntimeError, match="Final test set reuse blocked"):
        _record_final_test_evaluation(config, tmp_path / "run_b", trial_id="grid_000", split_hash="split", smoke=False)

    smoke = _record_final_test_evaluation(config, tmp_path / "run_c", trial_id="grid_000", split_hash="split", smoke=True)
    assert smoke["count"] == 3
