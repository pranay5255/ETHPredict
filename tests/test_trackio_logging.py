import json
import sys
import types
from datetime import datetime

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
