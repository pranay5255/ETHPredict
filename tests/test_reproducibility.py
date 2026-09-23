import json
from pathlib import Path

import pytest
import yaml

import src.experiments.meta_labeling_mvp as mvp
from src.experiments.meta_labeling_mvp import (
    _apply_selected_test_evaluations,
    _feature_manifest_with_identity,
    _git_manifest,
    _record_final_test_evaluation,
    check_raw_data_registry,
    feature_code_identity,
    research_spec_hash,
    select_trials_with_trade_floor,
    validate_selection_metric,
)
from src.experiments.reproducibility import compare_run_manifests
from src.experiments.staged_trial import run_staged_trial
from tests.test_staged_trial import _v2_config, _write_5m_ohlcv


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


def _research_config(**pipeline):
    return {
        "version": 2,
        "experiment": {"id": "spec", "name": "one", "seed": 7},
        "data": {"dir": "data", "granularity": "5m"},
        "features": {"include": ["ohlcv"]},
        "labels": {"meta_triple_barrier": {"profit_kappa": 1.0}},
        "model": {"base": {"hidden_size": 4}},
        "validation": {"folds": 2, "test_fraction": 0.2},
        "costs": {"fee_bps": 1.0},
        "alpha_backtest": {"meta_threshold": 0.55},
        "pipeline": {"selection_metric": "metrics.validation.net_pnl", "selection_mode": "max", "min_validation_trades": 1, **pipeline},
        "tracking": {"trackio": {"enabled": True}},
    }


def test_research_spec_hash_ignores_run_name_and_artifact_root_and_blocks_rename(tmp_path):
    ledger = tmp_path / "ledger.json"
    left = _research_config(run_name="alpha", artifact_root=str(tmp_path / "a"))
    right = _research_config(run_name="beta", artifact_root=str(tmp_path / "b"))
    right["experiment"] = {"id": "renamed", "name": "two", "seed": 7}
    right["tracking"] = {"trackio": {"enabled": False, "project": "elsewhere"}}
    left["research"] = {"final_test_guard": {"do_not_reuse_test": True, "ledger_path": str(ledger)}}
    right["research"] = {"final_test_guard": {"do_not_reuse_test": True, "ledger_path": str(ledger)}}

    assert research_spec_hash(left, split_hash="split", scope="meta_label_trial") == research_spec_hash(
        right, split_hash="split", scope="meta_label_trial"
    )

    seeded = _research_config()
    seeded["experiment"]["seed"] = 8
    assert research_spec_hash(seeded) != research_spec_hash(_research_config())

    first = _record_final_test_evaluation(left, tmp_path / "a" / "run", trial_id="grid_000", split_hash="split", smoke=False)
    assert first["count"] == 1
    with pytest.raises(RuntimeError, match="Final test set reuse blocked"):
        _record_final_test_evaluation(right, tmp_path / "b" / "run", trial_id="grid_001", split_hash="split", smoke=False)


def test_selection_uses_validation_metrics_and_rejects_test_metric():
    trials = [
        {
            "trial_id": "validation_winner",
            "metrics": {"validation": {"net_pnl": 5.0, "trades": 3}, "test": {"net_pnl": -50.0, "trades": 9}},
            "folds": [],
            "diagnostics": {},
        },
        {
            "trial_id": "test_winner",
            "metrics": {"validation": {"net_pnl": 1.0, "trades": 4}, "test": {"net_pnl": 80.0, "trades": 9}},
            "folds": [],
            "diagnostics": {},
        },
    ]
    config = {"pipeline": {"selection_metric": "metrics.validation.net_pnl", "selection_mode": "max", "min_validation_trades": 1}}

    raw_best, _, selection = select_trials_with_trade_floor(trials, config)

    assert raw_best["trial_id"] == "validation_winner"
    assert selection["metric"] == "metrics.validation.net_pnl"
    assert validate_selection_metric(config) == "metrics.validation.net_pnl"
    with pytest.raises(ValueError, match="final-test"):
        validate_selection_metric({"pipeline": {"selection_metric": "metrics.test.net_pnl"}})
    with pytest.raises(ValueError, match="final-test"):
        select_trials_with_trade_floor(trials, {"pipeline": {"selection_metric": "metrics.test.net_pnl"}})


def test_raw_data_change_warns_or_refuses_for_the_same_spec(tmp_path):
    root = tmp_path / "artifacts"
    original = {"files": [{"path": "data/raw/a.csv", "sha256": "aaa"}]}
    changed = {"files": [{"path": "data/raw/a.csv", "sha256": "bbb"}]}
    base = _research_config(run_name="first", artifact_root=str(root))
    renamed = _research_config(run_name="second", artifact_root=str(tmp_path / "elsewhere"))
    assert research_spec_hash(base, scope="raw_data") == research_spec_hash(renamed, scope="raw_data")

    failing = {**base, "research": {"raw_data_guard": {"on_change": "fail"}}}
    warning = {**renamed, "research": {"raw_data_guard": {"on_change": "warn"}}}

    first = check_raw_data_registry(failing, root, run_id="run-1", raw_manifest=original, smoke=False)
    assert first["status"] == "first_seen"
    with pytest.raises(RuntimeError, match="Raw data changed"):
        check_raw_data_registry(failing, root, run_id="run-2", raw_manifest=changed, smoke=False)
    with pytest.warns(RuntimeWarning, match="Raw data changed"):
        warned = check_raw_data_registry(warning, root, run_id="run-3", raw_manifest=changed, smoke=False)
    assert warned["status"] == "changed"
    assert warned["previous_run_id"] == "run-1"


def test_git_manifest_lists_critical_tooling_paths(monkeypatch):
    def fake_command(args):
        if list(args[:2]) == ["git", "status"]:
            return "\n".join(
                [
                    " M pyproject.toml",
                    "?? scripts/helper.py",
                    "?? uv.lock",
                    "?? notes.txt",
                    " M src/experiments/meta_labeling_mvp.py",
                ]
            )
        if list(args[:2]) == ["git", "rev-parse"]:
            return "abc123"
        return None

    monkeypatch.setattr(mvp, "_command_output", fake_command)
    manifest = _git_manifest()
    paths = {item["path"] for item in manifest["critical_untracked_or_modified"]}

    assert manifest["commit"] == "abc123"
    assert manifest["dirty"] is True
    assert {"pyproject.toml", "scripts/helper.py", "uv.lock", "src/experiments/meta_labeling_mvp.py"} <= paths
    assert "notes.txt" not in paths


def test_feature_manifest_includes_feature_code_identity(tmp_path):
    for relative in mvp.FEATURE_CODE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")

    first = feature_code_identity(tmp_path)
    assert first["hash"]
    assert set(first["files"]) == set(mvp.FEATURE_CODE_FILES)
    (tmp_path / "src/features/labeling.py").write_text("# changed\n", encoding="utf-8")
    assert feature_code_identity(tmp_path)["hash"] != first["hash"]

    manifest = _feature_manifest_with_identity(
        {"families": ["ohlcv"], "columns": ["close"], "fracdiff": {"mode": "fixed", "order": 0.4, "threshold": 0.01, "adf_pvalue": 0.2}}
    )
    assert manifest["code_identity"]["hash"]
    assert manifest["family_identity_hash"]


def test_unselected_trial_does_not_touch_the_final_test_ledger(tmp_path):
    selection = {
        "raw_best_trial_id": "grid_000",
        "best_trading_trial_id": "grid_000",
        "classification_best_trial_id": "grid_000",
        "calibration_best_trial_id": "grid_000",
    }
    trial = {
        "trial_id": "grid_001",
        "status": "completed",
        "metrics": {"validation": {"net_pnl": 0.0, "trades": 1}, "test": {"net_pnl": 99.0}},
        "manifest_path": tmp_path / "trials" / "grid_001" / "manifest.json",
    }
    config = {"research": {"final_test_guard": {"evaluate_selection_roles": ["raw_best"]}}}

    updated = _apply_selected_test_evaluations(
        [trial],
        selection,
        {},
        config=config,
        run_dir=tmp_path / "run",
        smoke=False,
        device=None,
    )

    assert updated[0]["test_evaluation"]["status"] == "not_selected"
    assert updated[0]["metrics"]["test"] == {}
    assert updated[0]["final_test_evaluation"] is None
    assert not (tmp_path / "run").parent.joinpath("_final_test_reuse_ledger.json").exists()
    saved = json.loads(Path(trial["manifest_path"]).read_text(encoding="utf-8"))
    assert saved["metrics"]["test"] == {}


def test_selected_trial_is_the_only_test_evaluation(tmp_path):
    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv")
    config = _v2_config(data_dir, tmp_path / "runs")
    config["benchmark"] = {"enabled": False}
    config["pipeline"]["min_validation_trades"] = 0
    config["pipeline"]["selection_metric"] = "metrics.validation.net_pnl"
    config["search"] = {
        "mode": "grid",
        "max_trials": 2,
        "trials": [
            {"id": "grid_000", "overrides": {}},
            {"id": "grid_001", "overrides": {"training.learning_rate": 0.002}},
        ],
    }
    config["smoke"]["max_trials"] = 2
    config["research"] = {"final_test_guard": {"do_not_reuse_test": True, "evaluate_selection_roles": ["raw_best"]}}
    config_path = tmp_path / "v2.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_staged_trial(config_path, smoke=True, device="cpu", allow_cpu=True)

    trials = {item["trial_id"]: item for item in result["trials"]}
    selected_id = result["selection"]["raw_best_trial_id"]
    selected = trials[selected_id]
    unselected = [item for trial_id, item in trials.items() if trial_id != selected_id]

    assert result["selection"]["metric"] == "metrics.validation.net_pnl"
    assert "test" not in result["selection"]["metric"].split(".")
    assert len(trials) == 2
    assert len(unselected) == 1
    assert selected["test_evaluation"]["status"] == "evaluated"
    assert selected["final_test_evaluation"]["count"] == 1
    assert "net_pnl" in selected["metrics"]["test"]
    assert selected["metrics"]["validation"]["net_pnl"] >= unselected[0]["metrics"]["validation"]["net_pnl"]
    assert unselected[0]["test_evaluation"]["status"] == "not_selected"
    assert unselected[0]["metrics"]["test"] == {}
    assert unselected[0]["final_test_evaluation"] is None
    assert "test_candidates" not in (unselected[0].get("artifact_paths") or {})

    ledger = json.loads((Path(result["run_dir"]).parent / "_final_test_reuse_ledger.json").read_text(encoding="utf-8"))
    meta_records = [record for record in ledger.values() if record.get("scope") == "meta_label_trial"]
    assert len(meta_records) == 1
    assert meta_records[0]["count"] == 1
    assert [item["trial_id"] for item in meta_records[0]["history"]] == [selected_id]
