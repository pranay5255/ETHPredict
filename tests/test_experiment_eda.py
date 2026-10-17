import json
from pathlib import Path

import pytest

from scripts import experiment_eda as eda


class FakeRun:
    def __init__(self, run_id, config, name="meta_label/example/grid_000"):
        self.id = run_id
        self.config = config
        self.name = name
        self.deleted = False

    def delete(self):
        self.deleted = True
        return True


def test_inventory_uses_metadata_and_preserves_research_named_smoke(monkeypatch):
    runs = [FakeRun("smoke-id", {"smoke": True}),
            FakeRun("setup-id", {"stage": "trackio_setup"}),
            FakeRun("research-id", {"smoke": False}, "research/smoke_word/grid_000")]

    class API:
        def runs(self, project):
            assert project == "ethpredict"
            return runs

    monkeypatch.setattr(eda, "_deps", lambda: (type("Trackio", (), {"Api": API}), None))
    groups = eda.inventory()
    assert {r.id for r in groups["smoke"]} == {"smoke-id"}
    assert {r.id for r in groups["setup"]} == {"setup-id"}
    assert {r.id for r in groups["research"]} == {"research-id"}


def test_cleanup_deletes_only_inventoried_ids_and_explicit_local_smoke(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    (artifacts / "runs" / "smoke_run").mkdir(parents=True)
    (artifacts / "runs" / "smoke_run" / "manifest.json").write_text(json.dumps({"smoke": True}))
    (artifacts / "runs" / "research_smoke_word").mkdir()
    (artifacts / "runs" / "research_smoke_word" / "manifest.json").write_text(json.dumps({"smoke": False}))
    (artifacts / "trackio_doctor").mkdir()
    receipt = artifacts / "trackio_doctor" / "setup_smoke_one.json"
    receipt.write_text(json.dumps({"run": "setup_smoke/one"}))
    smoke = FakeRun("s", {"smoke": True})
    setup = FakeRun("u", {"stage": "trackio_setup"}, "setup_smoke/one")
    research = FakeRun("r", {"smoke": False}, "research/smoke_word/grid_000")
    monkeypatch.setattr(eda, "ROOT", tmp_path)
    monkeypatch.setattr(eda, "ARTIFACTS", artifacts)
    monkeypatch.setattr(eda, "inventory", lambda: {"smoke": [] if smoke.deleted else [smoke],
                                                  "setup": [] if setup.deleted else [setup],
                                                  "research": [research]})
    eda.cleanup(True)
    assert smoke.deleted and setup.deleted and not research.deleted
    assert not (artifacts / "runs" / "smoke_run").exists()
    assert (artifacts / "runs" / "research_smoke_word").exists()
    assert not receipt.exists()
    audit = artifacts / "eda_cleanup_inventory.json"
    assert json.loads(audit.read_text())["before_ids"]["research"] == ["r"]
    before = audit.read_text()
    eda.cleanup(True)
    assert audit.read_text() == before


def test_normalization_missing_manifest_failed_and_zero_trades(tmp_path):
    missing = FakeRun("id1", {"stage": "meta_label_mvp_trial", "run_id": "v2_one",
                              "manifest_path": str(tmp_path / "gone.json"), "trackio_status": "failed"})
    row = eda.normalize(missing, {"alpha.validation.trades": 0, "alpha.test.trades": 7})
    assert row["status"] == "failed"
    assert row["evidence"] == "Trackio aggregate only"
    assert row["backtest"]["validation"]["trades"] == 0
    assert row["backtest"]["test"]["trades"] == 7
    assert row["artifact_available"]["manifest"] is False
    assert row["source_url"].endswith("/run?project=ethpredict&selected_run_id=id1")

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"status": "completed", "metrics": {"validation": {"trades": 0, "net_pnl": 0},
                                                               "test": {"trades": 5, "net_pnl": -4}},
                                "artifact_paths": {"oof_predictions": str(tmp_path / "missing.parquet")}}))
    local = FakeRun("id2", {"stage": "meta_label_mvp_trial", "run_id": "v2_two",
                            "manifest_path": str(path), "trackio_status": "success"})
    row = eda.normalize(local, {"alpha.validation.trades": 3, "alpha.validation.net_pnl": 10,
                                "alpha.test.trades": 5})
    assert row["status"] == "no trade"
    assert row["backtest"]["validation"]["net_pnl"] == 0
    assert row["backtest"]["test"]["net_pnl"] == -4
    assert len(row["discrepancies"]) == 2
    assert "oof_predictions" in row["missing_artifacts"]


def test_forecast_split_labels_remain_distinct():
    logs = {"forecast.validation.next_hour.return.mae": .1,
            "forecast.test.next_hour.return.mae": .9}
    rows = eda._forecast(None, logs)
    assert {(r["split"], r["mae"]) for r in rows} == {("validation", .1), ("test", .9)}


def test_local_metrics_override_trackio():
    local = {"metrics": {"validation": {"trades": 2, "net_pnl": 7},
                         "test": {"trades": 1, "net_pnl": -3}}}
    result = eda._backtest(local, {"alpha.validation.trades": 200,
                                   "alpha.validation.net_pnl": 900,
                                   "alpha.test.net_pnl": 800})
    assert result["validation"]["trades"] == 2
    assert result["validation"]["net_pnl"] == 7
    assert result["test"]["net_pnl"] == -3
