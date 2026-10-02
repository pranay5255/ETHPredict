import json
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from src.utils.trackio_logging import (
    TrackioLoggingError,
    enforce_trackio_policy,
    flatten_numeric,
    json_ready,
    log_trackio_run,
    trackio_enabled,
)


def test_flatten_numeric_keeps_only_finite_scalars():
    payload = {
        "forecast": {
            "mae": np.float32(1.25),
            "count": np.int64(7),
            "ok": True,
            "bad": float("nan"),
            "table": [{"bucket": "x", "count": 1}],
        },
        "label": "not_numeric",
    }

    assert flatten_numeric(payload) == {
        "forecast.mae": 1.25,
        "forecast.count": 7,
        "forecast.ok": 1.0,
    }


def test_json_ready_converts_paths_numpy_and_timestamps(tmp_path):
    payload = {
        "path": tmp_path / "predictions.parquet",
        "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
        "datetime": datetime(2026, 1, 1, 12, 30),
        "array": np.array([1, 2]),
        "finite": np.float64(2.5),
        "bad": float("inf"),
        "flag": np.bool_(True),
    }

    ready = json_ready(payload)

    assert ready["path"] == str(tmp_path / "predictions.parquet")
    assert ready["timestamp"].startswith("2026-01-01T00:00:00")
    assert ready["datetime"] == "2026-01-01T12:30:00"
    assert ready["array"] == [1, 2]
    assert ready["finite"] == 2.5
    assert ready["bad"] is None
    assert ready["flag"] is True
    json.dumps(ready)


def test_log_trackio_run_uses_config_flattens_metrics_and_finishes(monkeypatch, tmp_path):
    calls = {"init": [], "log": [], "finish": 0}

    def fake_init(**kwargs):
        calls["init"].append(kwargs)

    def fake_log(metrics):
        calls["log"].append(metrics)

    def fake_finish():
        calls["finish"] += 1

    fake_trackio = types.SimpleNamespace(init=fake_init, log=fake_log, finish=fake_finish)
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)
    config = {
        "tracking": {
            "trackio": {
                "enabled": True,
                "project": "ethpredict",
                "auto_log_gpu": True,
                "gpu_log_interval": 10.0,
                "auto_log_cpu": False,
            }
        }
    }

    logged = log_trackio_run(
        config,
        name="benchmark/run_123/momentum",
        group="forecast_benchmark",
        run_config={"model_id": "momentum", "status": "success"},
        metrics={"forecast": {"mae": np.float64(0.1), "buckets": [{"count": 1}]}, "backtest": {"net_pnl": 3.0}},
        artifacts={"predictions": tmp_path / "predictions.parquet"},
        smoke=True,
        receipt_path=tmp_path / "receipt.json",
    )

    assert logged is True
    assert calls["finish"] == 1
    assert calls["init"][0]["project"] == "ethpredict"
    assert calls["init"][0]["name"] == "benchmark/run_123/momentum"
    assert calls["init"][0]["group"] == "forecast_benchmark"
    assert calls["init"][0]["auto_log_gpu"] is True
    assert calls["init"][0]["gpu_log_interval"] == 10.0
    assert calls["init"][0]["auto_log_cpu"] is False
    assert calls["init"][0]["config"]["artifact_paths"]["predictions"] == str(tmp_path / "predictions.parquet")
    assert calls["log"][0] == {
        "forecast.mae": 0.1,
        "backtest.net_pnl": 3.0,
        "trackio.status.success": 1.0,
    }
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["delivery"] == "logged"
    assert receipt["verification"] == "not_required"


def test_required_trackio_readback_failure_is_recorded_and_raises(monkeypatch, tmp_path):
    calls = {"finish": 0}

    def finish():
        calls["finish"] += 1

    monkeypatch.setitem(sys.modules, "trackio", types.SimpleNamespace(init=lambda **_: None, log=lambda _: None, finish=finish))
    monkeypatch.setattr("src.utils.trackio_logging._verify_local_run", lambda *_: (_ for _ in ()).throw(ValueError("missing metrics")))
    config = {"tracking": {"trackio": {"enabled": True, "project": "ethpredict"}}}
    receipt_path = tmp_path / "receipt.json"

    with pytest.raises(TrackioLoggingError, match="missing metrics"):
        log_trackio_run(config, name="run/check", group="research", receipt_path=receipt_path)

    assert calls["finish"] == 1
    assert json.loads(receipt_path.read_text())["delivery"] == "failed"


def test_required_trackio_init_failure_raises_but_debug_override_allows_it(monkeypatch, tmp_path):
    def fail_init(**_):
        raise OSError("local store unavailable")

    monkeypatch.setitem(sys.modules, "trackio", types.SimpleNamespace(init=fail_init, log=lambda _: None, finish=lambda: None))
    config = {"tracking": {"trackio": {"enabled": True}}}
    with pytest.raises(TrackioLoggingError, match="local store unavailable"):
        log_trackio_run(config, name="run/check", group="research", receipt_path=tmp_path / "required.json")
    assert json.loads((tmp_path / "required.json").read_text())["delivery"] == "failed"

    config["tracking"]["trackio"]["allow_local_debug_without_trackio"] = True
    with pytest.warns(RuntimeWarning, match="local store unavailable"):
        assert not log_trackio_run(config, name="run/debug", group="research")


def test_trackio_disabled_by_default():
    assert trackio_enabled({}) is False


def test_trackio_policy_requires_tracking_for_non_smoke_runs():
    with pytest.raises(RuntimeError, match="Trackio is required"):
        enforce_trackio_policy({}, smoke=False)

    enforce_trackio_policy({}, smoke=True)
    enforce_trackio_policy({"tracking": {"trackio": {"enabled": True}}}, smoke=False)
    enforce_trackio_policy({"tracking": {"trackio": {"allow_local_debug_without_trackio": True}}}, smoke=False)


def _capture_trackio(monkeypatch):
    calls = []

    def capture(config, **kwargs):
        calls.append(kwargs)
        receipt = kwargs.get("receipt_path")
        if receipt is not None:
            Path(receipt).parent.mkdir(parents=True, exist_ok=True)
            Path(receipt).write_text(json.dumps({"delivery": "logged", "run": kwargs.get("name")}), encoding="utf-8")
        return True

    monkeypatch.setattr("src.experiments.meta_labeling_mvp.log_trackio_run", capture)
    return calls


def test_schema_accepts_trackio_local_debug_override():
    import jsonschema
    import yaml

    schema = yaml.safe_load(Path("configs/schema.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load(Path("configs/config.yml").read_text(encoding="utf-8"))
    config["tracking"]["trackio"]["allow_local_debug_without_trackio"] = True

    jsonschema.validate(instance=config, schema=schema)


def test_trackio_payload_uses_persisted_failure_modes_and_reasons(monkeypatch, tmp_path):
    from src.experiments.meta_labeling_mvp import (
        _failed_trial_manifest,
        _log_meta_label_trial_trackio,
        _skipped_trial_manifest,
        _trial_accounting,
    )

    calls = _capture_trackio(monkeypatch)
    both = {
        "trial_id": "both",
        "status": "completed",
        "metrics": {
            "validation": {"trades": 0, "net_pnl": 1.0, "coverage": 0.5, "turnover": 0.0, "fees": 0.0, "gross_pnl": 0.0},
            "test": {},
        },
        "diagnostics": {"validation": {}},
        "folds": [],
        "manifest_path": tmp_path / "both" / "manifest.json",
        "config": {"costs": {"fee_bps": 1.0}, "alpha_backtest": {"meta_threshold": 0.55}},
        "dataset": {"feature_manifest": {"families": ["ohlcv"]}},
        "split_manifest_hash": "split-hash",
        "checkpoint_id": "checkpoint",
    }
    other = {
        "trial_id": "other",
        "status": "completed",
        "metrics": {
            "validation": {"trades": 3, "net_pnl": -1.0, "coverage": 0.2, "turnover": 10.0, "fees": 1.0, "gross_pnl": 2.0},
            "test": {"net_pnl": 1.0, "trades": 1},
        },
        "diagnostics": {"validation": {}},
        "folds": [],
    }
    selection = {
        "min_validation_trades": 1,
        "raw_best_trial_id": "other",
        "best_trading_trial_id": "other",
        "classification_best_trial_id": "both",
        "calibration_best_trial_id": "both",
    }
    accounting = _trial_accounting([both, other], selection, {"pipeline": {"min_validation_trades": 1}, "alpha_backtest": {"initial_capital": 10000}})
    persisted = accounting["trials"][0]["failure_modes"]
    assert persisted["flags"]["no_trade_winner"] is True

    _log_meta_label_trial_trackio(
        {},
        run_id="run",
        trial=both,
        selection_status="classification_calibration_best",
        raw_best={"trial_id": "other"},
        best_trading={"trial_id": "other"},
        selection=selection,
        trial_count=2,
        smoke=True,
    )

    logged = calls[0]
    assert logged["metrics"]["trial_accounting"]["failure_modes"]["flags"] == persisted["flags"]
    assert logged["metrics"]["trial_accounting"]["failure_modes"] is persisted
    assert logged["run_config"]["feature_families"] == ["ohlcv"]
    assert logged["run_config"]["split_manifest_hash"] == "split-hash"
    assert logged["run_config"]["costs"]["fee_bps"] == 1.0
    assert logged["run_config"]["alpha_policy"]["meta_threshold"] == 0.55
    assert logged["run_config"]["trial_count"] == 2
    assert logged["run_config"]["checkpoint_id"] == "checkpoint"

    failed = _failed_trial_manifest(
        {"trial_id": "failed_one", "trial_index": 1, "config": {}},
        tmp_path,
        smoke=True,
        error=RuntimeError("model diverged"),
    )
    skipped = _skipped_trial_manifest(
        {"trial_id": "skipped_one", "trial_index": 0, "config": {}},
        tmp_path,
        smoke=True,
        reason="held out",
    )
    _trial_accounting([failed, skipped], {"min_validation_trades": 1}, {})
    _log_meta_label_trial_trackio(
        {},
        run_id="run",
        trial=failed,
        selection_status="failed",
        raw_best={},
        best_trading=None,
        selection={"min_validation_trades": 1},
        trial_count=2,
        smoke=True,
    )
    _log_meta_label_trial_trackio(
        {},
        run_id="run",
        trial=skipped,
        selection_status="skipped",
        raw_best={},
        best_trading=None,
        selection={"min_validation_trades": 1},
        trial_count=2,
        smoke=True,
    )

    assert calls[1]["run_config"]["reason"] == "model diverged"
    assert calls[1]["run_config"]["skip_reason"] == "model diverged"
    assert calls[2]["run_config"]["reason"] == "held out"
    assert calls[2]["run_config"]["skip_reason"] == "held out"


def test_all_failed_trials_log_accounting_before_raising(monkeypatch, tmp_path):
    import yaml

    from src.experiments.meta_labeling_mvp import run_meta_labeling_mvp
    from tests.test_staged_trial import _v2_config, _write_5m_ohlcv

    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", rows=16)
    artifact_root = tmp_path / "runs"
    config = _v2_config(data_dir, artifact_root)
    config["benchmark"] = {"enabled": False}
    config["search"] = {
        "mode": "grid",
        "max_trials": 2,
        "trials": [
            {"id": "skipped_one", "overrides": {"trial.skip": True, "trial.skip_reason": "held out"}},
            {"id": "failed_one", "overrides": {}},
        ],
    }
    config["smoke"]["max_trials"] = 2
    config_path = tmp_path / "v2.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    calls = _capture_trackio(monkeypatch)

    def fail_trial(*args, **kwargs):
        raise RuntimeError("model diverged")

    monkeypatch.setattr("src.experiments.meta_labeling_mvp.build_multi_horizon_lighter_dataset", lambda *args, **kwargs: {})
    monkeypatch.setattr("src.experiments.meta_labeling_mvp._write_stage0", lambda dataset, run_dir: {"stage": "stage0_features"})
    monkeypatch.setattr("src.experiments.meta_labeling_mvp._run_one_trial", fail_trial)

    with pytest.raises(RuntimeError, match="All v2 trial specs failed"):
        run_meta_labeling_mvp(config_path, smoke=True, device="cpu", allow_cpu=True)

    assert calls
    assert calls[0]["name"].endswith("/accounting")
    assert calls[0]["run_config"]["stage"] == "meta_label_mvp_accounting"
    reasons = {call["run_config"].get("trial_id"): call["run_config"].get("skip_reason") for call in calls[1:]}
    assert reasons["skipped_one"] == "held out"
    assert reasons["failed_one"] == "model diverged"
    accounting_files = list(artifact_root.rglob("trial_accounting.json"))
    assert len(accounting_files) == 1
    accounting = json.loads(accounting_files[0].read_text(encoding="utf-8"))
    receipts = {row["trial_id"]: str(row["trackio_receipt_path"]) for row in accounting["trials"]}
    assert receipts["failed_one"].endswith("trackio_receipt.json")
    assert receipts["skipped_one"].endswith("trackio_receipt.json")
    assert str(accounting["summary_trackio_receipt_path"]).endswith("tracking/accounting.json")
    failed_manifest = json.loads(next(artifact_root.rglob("trials/failed_one/manifest.json")).read_text(encoding="utf-8"))
    failed_row = next(row for row in accounting["trials"] if row["trial_id"] == "failed_one")
    assert failed_manifest["reason"] == "model diverged"
    assert failed_manifest["skip_reason"] == "model diverged"
    assert failed_manifest["trackio_receipt_path"].endswith("trackio_receipt.json")
    assert failed_manifest["failure_modes"]["flags"] == failed_row["failure_modes"]["flags"]


def test_forecast_benchmark_requires_trackio_when_not_smoke(tmp_path):
    import yaml

    from src.experiments.forecast_benchmark import run_forecast_benchmark

    config_path = tmp_path / "benchmark.yml"
    config_path.write_text(yaml.safe_dump({"version": 2, "pipeline": {"run_name": "bench"}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Trackio is required"):
        run_forecast_benchmark(config_path, smoke=False, allow_cpu=True)
