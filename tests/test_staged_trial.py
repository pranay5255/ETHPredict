import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.experiments.meta_labeling_mvp import (
    _run_trial_safe,
    _trial_accounting,
    build_multi_horizon_lighter_dataset,
    fit_meta_labeler,
    purged_walk_forward_splits,
    run_alpha_backtest,
    train_base_model,
)
from src.config.loader import expand_grid_search
from src.experiments.staged_trial import run_staged_trial
from src.features.labeling import meta_triple_barrier_labels


def _write_5m_ohlcv(path: Path, rows: int = 96, step: float = 0.001):
    interval_ms = 300_000
    start_ms = 1774656000000
    closes = 3000.0 * np.exp(np.arange(rows) * step)
    frame = pd.DataFrame(
        {
            "open_time": [start_ms + i * interval_ms for i in range(rows)],
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": [10.0 + i for i in range(rows)],
            "close_time": [start_ms + i * interval_ms + interval_ms - 1 for i in range(rows)],
            "quote_asset_volume": closes * (10.0 + np.arange(rows)),
            "number_of_trades": [0 for _ in range(rows)],
            "taker_buy_base_asset_volume": [0.0 for _ in range(rows)],
            "taker_buy_quote_asset_volume": [0.0 for _ in range(rows)],
            "ignore": [0 for _ in range(rows)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, header=False, index=False)
    return closes


def test_staged_trial_writes_model_backtest_and_report_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv")
    config_path = tmp_path / "staged.yml"
    config = {
        "data": {"dir": str(data_dir), "granularity": "5m"},
        "targets": ["next_hour_return"],
        "labels": {"kappa": 2.0, "timeout": 8, "volatility_window": 8, "next_hour_horizon": 2},
        "training": {
            "sequence_length": 8,
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.001,
            "neural_trials_per_target": 1,
        },
        "smoke": {"max_rows": 96, "epochs": 1, "batch_size": 8, "hidden_size": 4},
        "glft": {
            "initial_capital": 100000.0,
            "gamma": 0.5,
            "kappa": 0.1,
            "dt": 1 / 288,
            "max_inventory": 10.0,
            "min_spread": 5.0,
            "max_drawdown": 0.2,
            "seed": 7,
        },
        "staged_trial": {
            "artifact_root": str(tmp_path / "runs"),
            "run_name": "pytest_staged",
            "target": "next_hour_return",
            "selection_metric": "metrics.validation.prediction.mae",
            "selection_mode": "min",
            "split": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2},
            "stage1": {
                "max_trials": 2,
                "prediction_batch_size": 16,
                "search_space": {"hidden_size": [4, 6], "learning_rate": [0.001]},
            },
            "stage2": {
                "max_rows": 20,
                "selection_metric": "metrics.net_pnl",
                "selection_mode": "max",
                "max_trials": 2,
                "strategy_search": {"gamma": [0.3, 0.5], "kappa": [0.1], "min_spread": [5.0]},
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_staged_trial(config_path, smoke=True, device="cpu", allow_cpu=True)

    report_path = Path(result["report_json_path"])
    report = json.loads(report_path.read_text())
    assert report["status"] == "success"
    assert Path(report["stage0_manifest_path"]).exists()
    assert len(report["stage1"]["trials"]) == 2
    assert len(report["stage2"]["strategies"]) == 2

    best_model = json.loads(Path(report["stage1"]["best_model_manifest_path"]).read_text())
    assert Path(best_model["model_paths"]["hierarchical"]).exists()
    assert Path(best_model["prediction_paths"]["test"]).exists()

    best_strategy = json.loads(Path(report["stage2"]["best_strategy_manifest_path"]).read_text())
    assert Path(best_strategy["consumed_predictions"]).resolve() == Path(best_model["prediction_paths"]["test"]).resolve()
    assert Path(best_strategy["trades_path"]).exists()
    assert Path(result["report_markdown_path"]).exists()


def _v2_config(data_dir: Path, artifact_root: Path) -> dict:
    return {
        "version": 2,
        "experiment": {"id": "pytest_v2", "seed": 11, "trials": 1},
        "data": {"dir": str(data_dir), "granularity": "5m", "sources": ["lighter"]},
        "targets": {"horizons": {"next_5m": {"bars": 1}, "next_hour": {"bars": 3}}},
        "validation": {"method": "purged_walk_forward", "folds": 2, "test_fraction": 0.2, "purge_bars": 2, "embargo_bars": 2},
        "labels": {
            "meta_triple_barrier": {
                "profit_kappa": 1.0,
                "stop_kappa": 1.0,
                "vertical_barriers": {"next_5m": 1, "next_hour": 3},
                "volatility_window": 8,
                "min_edge_bps": 0.0,
                "cost_mode": "configured",
            }
        },
        "model": {
            "base": {"type": "lstm", "hidden_size": 4, "num_layers": 1, "dropout": 0.0},
            "meta_labeler": {"type": "mlp", "hidden_size": 4, "num_layers": 1, "dropout": 0.0, "enabled": True},
        },
        "training": {"sequence_length": 8, "epochs": 1, "batch_size": 8, "learning_rate": 0.001, "loss_weights": {"return": 1.0, "direction": 0.1}},
        "costs": {"fee_bps": 1.0, "spread_bps": 1.0, "slippage_bps": 0.0, "funding_bps_per_hour": 0.0},
        "alpha_backtest": {"initial_capital": 10000.0, "horizon": "best", "meta_threshold": 0.5, "edge_threshold_bps": 0.0, "position_notional": 1000.0},
        "pipeline": {"artifact_root": str(artifact_root), "run_name": "pytest_v2"},
        "search": {"mode": "grid", "max_trials": 1, "spaces": {}},
        "smoke": {"max_rows": 96, "max_trials": 1, "folds": 2, "sequence_length": 8, "epochs": 1, "hidden_size": 4, "batch_size": 8},
    }


def test_multi_horizon_dataset_aligns_targets_without_future_features(tmp_path):
    data_dir = tmp_path / "data"
    closes = _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", rows=40, step=0.01)
    config = _v2_config(data_dir, tmp_path / "runs")
    config["training"]["sequence_length"] = 2

    dataset = build_multi_horizon_lighter_dataset(config, smoke=False)

    assert dataset["horizon_names"] == ["next_5m", "next_hour"]
    assert int(dataset["samples"].iloc[0]["sample_index"]) == 2
    assert np.isclose(float(dataset["y_ret"][0, 0]), np.log(closes[3] / closes[2]))
    assert np.isclose(float(dataset["y_ret"][0, 1]), np.log(closes[5] / closes[2]))
    assert int(dataset["samples"]["sample_index"].max()) + dataset["max_horizon_bars"] <= len(dataset["price_path"]) - 1


def test_purged_walk_forward_splits_keep_chronology_purge_embargo_and_test_gap():
    splits = purged_walk_forward_splits(
        100,
        {"folds": 2, "test_fraction": 0.2, "purge_bars": 5, "embargo_bars": 3},
    )

    assert splits["test"][0] == 80
    assert splits["gap"][0] == 72
    assert splits["gap"][-1] == 79
    for fold in splits["folds"]:
        assert fold["train"][-1] <= fold["validation"][0] - 6
        assert fold["validation"][-1] < splits["test"][0]


def test_meta_triple_barrier_labels_profit_stop_vertical_and_no_signal():
    path = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 101.2, 100.5],
            "low": [100.0, 99.8, 99.5],
            "close": [100.0, 100.8, 101.0],
        }
    )
    base = {"sample_index": 0, "side": 1, "horizon_bars": 2, "realized_vol": 0.01, "total_cost_bps": 0.0}
    signals = pd.DataFrame(
        [
            {**base, "is_candidate": True},
            {**base, "is_candidate": False},
        ]
    )
    labeled = meta_triple_barrier_labels(signals, path, {"profit_kappa": 1.0, "stop_kappa": 1.0, "min_edge_bps": 0}, {}, granularity="5m")
    assert labeled.iloc[0]["exit_reason"] == "profit_take"
    assert labeled.iloc[0]["meta_label"] == 1.0
    assert int(labeled.iloc[0]["vertical_barrier_idx"]) == 2
    assert int(labeled.iloc[0]["t1_idx"]) == 1
    assert int(labeled.iloc[0]["label_span_end_idx"]) == 1
    assert np.isnan(labeled.iloc[1]["meta_label"])
    assert int(labeled.iloc[1]["vertical_barrier_idx"]) == 2
    assert int(labeled.iloc[1]["t1_idx"]) == 2
    assert int(labeled.iloc[1]["label_span_end_idx"]) == 2

    stop_path = path.copy()
    stop_path.loc[1, "high"] = 100.2
    stop_path.loc[1, "low"] = 98.0
    stopped = meta_triple_barrier_labels(pd.DataFrame([{**base, "is_candidate": True}]), stop_path, {"profit_kappa": 1.0, "stop_kappa": 1.0}, {}, granularity="5m")
    assert stopped.iloc[0]["exit_reason"] == "stop_loss"
    assert stopped.iloc[0]["meta_label"] == 0.0
    assert int(stopped.iloc[0]["t1_idx"]) == 1
    assert int(stopped.iloc[0]["label_span_end_idx"]) == 1

    vertical = meta_triple_barrier_labels(pd.DataFrame([{**base, "is_candidate": True, "realized_vol": 0.10}]), path, {"profit_kappa": 2.0, "stop_kappa": 2.0}, {}, granularity="5m")
    assert vertical.iloc[0]["exit_reason"] == "vertical"
    assert vertical.iloc[0]["meta_label"] == 1.0
    assert int(vertical.iloc[0]["t1_idx"]) == 2
    assert int(vertical.iloc[0]["label_span_end_idx"]) == 2

    incomplete = meta_triple_barrier_labels(
        pd.DataFrame([{**base, "sample_index": 2, "is_candidate": True}]),
        path,
        {"profit_kappa": 1.0, "stop_kappa": 1.0},
        {},
        granularity="5m",
    )
    assert incomplete.iloc[0]["exit_reason"] == "incomplete_path"
    assert int(incomplete.iloc[0]["t1_idx"]) == 2
    assert int(incomplete.iloc[0]["label_span_end_idx"]) == 2


def test_alpha_backtest_has_explicit_cost_accounting():
    candidates = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01"),
                "horizon": "next_5m",
                "side": 1,
                "close": 100.0,
                "is_candidate": True,
                "meta_prob": 0.8,
                "expected_edge_bps": 20.0,
                "total_cost_bps": 10.0,
                "label_gross_return": 0.01,
                "label_net_return": 0.009,
            },
            {
                "timestamp": pd.Timestamp("2026-01-02"),
                "horizon": "next_5m",
                "side": 1,
                "close": 100.0,
                "is_candidate": True,
                "meta_prob": 0.4,
                "expected_edge_bps": 20.0,
                "total_cost_bps": 10.0,
                "label_gross_return": 0.01,
                "label_net_return": 0.009,
            },
        ]
    )
    metrics, trades = run_alpha_backtest(
        candidates,
        {"alpha_backtest": {"initial_capital": 10000.0, "meta_threshold": 0.5, "edge_threshold_bps": 5.0, "position_notional": 1000.0, "horizon": "best"}},
    )

    assert len(trades) == 1
    assert metrics["gross_pnl"] == 10.0
    assert metrics["fees"] == 1.0
    assert metrics["net_pnl"] == 9.0


def test_event_sampling_can_filter_observation_times(tmp_path):
    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", rows=48, step=0.01)
    config = _v2_config(data_dir, tmp_path / "runs")
    config["training"]["sequence_length"] = 4
    config["sampling"] = {"mode": "cusum", "cusum_threshold_bps": 1.0}

    dataset = build_multi_horizon_lighter_dataset(config, smoke=False)

    assert dataset["event_diagnostics"]["mode"] == "cusum"
    assert 0 < dataset["event_diagnostics"]["events"] <= dataset["event_diagnostics"]["dense_observations"]
    assert dataset["event_diagnostics"]["average_uniqueness"] > 0.0
    assert dataset["event_diagnostics"]["effective_sample_size"] > 0.0
    assert "event_time" in dataset["samples"].columns
    assert set(dataset["samples"]["event_trigger_type"]).issubset({"positive_cusum", "negative_cusum"})

    config["sampling"] = {"mode": "volatility_cusum", "volatility_multiplier": 0.1, "min_threshold_bps": 1.0}
    vol_dataset = build_multi_horizon_lighter_dataset(config, smoke=False)
    assert vol_dataset["event_diagnostics"]["mode"] == "volatility_cusum"


def test_edge_triggered_sampling_uses_configured_feature_column(tmp_path):
    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", rows=48, step=0.01)
    config = _v2_config(data_dir, tmp_path / "runs")
    config["training"]["sequence_length"] = 4
    config["sampling"] = {"mode": "edge_triggered", "edge_column": "log_return", "edge_threshold_bps": 1.0}

    dataset = build_multi_horizon_lighter_dataset(config, smoke=False)

    assert dataset["event_diagnostics"]["mode"] == "edge_triggered"
    assert dataset["event_diagnostics"]["edge_column"] == "log_return"
    assert set(dataset["samples"]["event_trigger_type"]) == {"edge_triggered"}


def test_explicit_v2_search_trials_keep_ids_and_overrides(tmp_path):
    config = _v2_config(tmp_path / "data", tmp_path / "runs")
    config["search"] = {
        "mode": "grid",
        "max_trials": 2,
        "trials": [
            {"id": "normal", "overrides": {"alpha_backtest.meta_threshold": 0.4}},
            {"id": "skip_case", "overrides": {"trial.skip": True, "trial.skip_reason": "fixture skip"}},
            {"id": "ignored", "overrides": {}},
        ],
    }

    specs = expand_grid_search(config)

    assert [spec["trial_id"] for spec in specs] == ["normal", "skip_case"]
    assert specs[0]["config"]["alpha_backtest"]["meta_threshold"] == 0.4
    assert specs[1]["config"]["trial"]["skip"] is True


def test_skipped_v2_trial_writes_manifest_and_accounting(tmp_path):
    config = _v2_config(tmp_path / "data", tmp_path / "runs")
    spec = {
        "trial_index": 0,
        "trial_id": "skip_case",
        "overrides": {"trial.skip": True},
        "config": {**config, "trial": {"skip": True, "skip_reason": "fixture skip"}},
    }

    manifest = _run_trial_safe(spec, tmp_path / "runs" / "skip_trial", smoke=False, device=torch.device("cpu"))

    assert manifest["status"] == "skipped"
    assert manifest["skip_reason"] == "fixture skip"
    assert Path(manifest["manifest_path"]).exists()
    accounting = _trial_accounting([manifest], {"min_validation_trades": 1}, config)
    assert accounting["counts"]["skipped"] == 1
    assert accounting["counts"]["no_trade"] == 0
    assert accounting["counts"]["low_trade"] == 0


def test_failed_v2_trial_writes_manifest_with_reason(tmp_path):
    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", rows=48, step=0.01)
    config = _v2_config(data_dir, tmp_path / "runs")
    config["training"]["sequence_length"] = 4
    config["sampling"] = {"mode": "edge_triggered", "edge_column": "missing_edge", "edge_threshold_bps": 1.0}
    spec = {"trial_index": 0, "trial_id": "bad_sampling", "overrides": {}, "config": config}

    manifest = _run_trial_safe(spec, tmp_path / "runs" / "safe_trial", smoke=False, device=torch.device("cpu"))

    assert manifest["status"] == "failed"
    assert manifest["skip_reason"]
    assert manifest["error_type"] == "ValueError"
    assert "missing_edge" in manifest["reason"]
    assert Path(manifest["manifest_path"]).exists()

    accounting = _trial_accounting([manifest], {"min_validation_trades": 1}, config)
    assert accounting["counts"]["failed"] == 1
    assert accounting["counts"]["no_trade"] == 0
    assert accounting["counts"]["low_trade"] == 0


def test_base_model_scaler_is_fit_on_train_indices_only(tmp_path):
    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", rows=48, step=0.002)
    config = _v2_config(data_dir, tmp_path / "runs")
    config["training"]["sequence_length"] = 4
    dataset = build_multi_horizon_lighter_dataset(config, smoke=False)
    train_indices = np.arange(0, min(12, int(dataset["X"].shape[0])), dtype=int)

    model, history = train_base_model(dataset, train_indices, config, torch.device("cpu"))

    expected_mean = dataset["X"][train_indices].mean(dim=(0, 1), keepdim=True)
    full_mean = dataset["X"].mean(dim=(0, 1), keepdim=True)
    assert torch.allclose(model.feature_mean, expected_mean)
    assert not torch.allclose(model.feature_mean, full_mean)
    assert history["preprocessing"]["fit_scope"] == "train_indices_only"


def test_v2_dataset_emits_span_metadata_and_non_uniform_configured_weights(tmp_path):
    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", rows=64, step=0.002)
    config = _v2_config(data_dir, tmp_path / "runs")
    config["training"]["sequence_length"] = 4
    config["sample_weights"] = {
        "base_mode": "horizon_span_uniqueness",
        "meta_mode": "triple_barrier_t1_uniqueness",
        "normalize": "sum_one",
    }

    dataset = build_multi_horizon_lighter_dataset(config, smoke=False)
    samples = dataset["samples"]
    required_columns = {
        "label_span_start_idx",
        "label_span_end_idx",
        "label_span_start_timestamp",
        "label_span_end_timestamp",
        "next_5m_label_span_end_idx",
        "next_5m_label_span_end_timestamp",
        "next_hour_label_span_end_idx",
        "next_hour_label_span_end_timestamp",
        "average_uniqueness",
        "sample_weight",
    }

    assert required_columns.issubset(samples.columns)
    assert np.isclose(float(dataset["sample_weights"].sum()), 1.0)
    assert not np.allclose(dataset["sample_weights"].numpy(), np.full(len(samples), 1.0 / len(samples)))
    assert dataset["sample_weight_diagnostics"]["base_mode"] == "horizon_span_uniqueness"
    assert dataset["sample_weight_diagnostics"]["concurrency"]["max"] > 1.0
    assert dataset["sample_weight_diagnostics"]["weights"]["sum"] == 1.0

    train_indices = np.arange(0, min(12, int(dataset["X"].shape[0])), dtype=int)
    model, history = train_base_model(dataset, train_indices, config, torch.device("cpu"))

    assert model.input_size == dataset["input_size"]
    assert np.isfinite(history["final_loss"])


def test_meta_labeler_training_accepts_non_uniform_candidate_weights():
    candidates = pd.DataFrame(
        {
            "is_candidate": [True, True, True, True],
            "meta_label": [1.0, 0.0, 1.0, 0.0],
            "sample_weight": [0.05, 0.45, 0.10, 0.40],
            "pred_return": [0.01, -0.01, 0.02, -0.02],
            "direction_prob": [0.8, 0.2, 0.7, 0.3],
            "direction_confidence": [0.8, 0.8, 0.7, 0.7],
            "expected_edge_bps": [50.0, 30.0, 60.0, 20.0],
            "total_cost_bps": [2.0, 2.0, 2.0, 2.0],
            "realized_vol": [0.01, 0.01, 0.02, 0.02],
            "vol_regime": [0.0, 0.0, 1.0, 1.0],
            "horizon_bars": [3, 3, 3, 3],
            "side": [1, -1, 1, -1],
            "horizon_disagreement": [0.0, 0.0, 1.0, 1.0],
            "next_5m_pred_return": [0.01, -0.01, 0.02, -0.02],
            "next_5m_direction_prob": [0.8, 0.2, 0.7, 0.3],
        }
    )
    config = {
        "model": {"meta_labeler": {"hidden_size": 4, "num_layers": 1, "dropout": 0.0}},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
    }

    model = fit_meta_labeler(candidates, config, ["next_5m"], torch.device("cpu"))

    assert model.manifest()["kind"] == "mlp"


def test_v2_staged_trial_runs_meta_labeling_alpha_smoke(tmp_path):
    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv", rows=96, step=0.002)
    config_path = tmp_path / "v2.yml"
    config_path.write_text(yaml.safe_dump(_v2_config(data_dir, tmp_path / "runs")), encoding="utf-8")

    result = run_staged_trial(config_path, smoke=True, device="cpu", allow_cpu=True)

    assert result["status"] == "success"
    assert result["best_trial"]["trial_id"] == "grid_000"
    assert Path(result["best_trial_manifest_path"]).exists()
    assert Path(result["best_trial"]["artifact_paths"]["test_candidates"]).exists()
    assert result["raw_data"]["files"][0]["sha256"]
    assert result["run_identity"]["resolved_config_hash"]
    assert result["trial_accounting"]["trial_count"] == 1
    assert result["selection"]["classification_best_trial_id"] == "grid_000"
    assert result["selection"]["calibration_best_trial_id"] == "grid_000"
    assert "classification_best" in result["trial_accounting"]["trials"][0]["selection_roles"]
    assert "calibration_best" in result["trial_accounting"]["trials"][0]["selection_roles"]

    trial_manifest = json.loads(Path(result["best_trial_manifest_path"]).read_text(encoding="utf-8"))
    checkpoint_path = Path(trial_manifest["model_path"])
    assert trial_manifest["checkpoint_id"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert json.loads((checkpoint_path.parent / "trackio_receipt.json").read_text())["delivery"] == "disabled"
    assert trial_manifest["split_manifest_hash"]
    assert trial_manifest["final_test_evaluation"]["count"] >= 1
    assert trial_manifest["test_evaluation"]["status"] == "evaluated"
    assert "net_pnl" in trial_manifest["metrics"]["test"]
    assert trial_manifest["dataset"]["feature_manifest"]["code_identity"]["hash"]
    assert result["run_identity"]["research_spec_hash"]
    assert result["run_identity"]["raw_data_guard"]["status"] in {"first_seen", "unchanged", "changed"}
    assert trial_manifest["dataset"]["feature_manifest"]["columns"]
    assert result["stage0"]["sample_weight_diagnostics"]["base_mode"] == "uniform"
    assert np.isclose(result["stage0"]["sample_weight_diagnostics"]["weights"]["sum"], 1.0)
    assert trial_manifest["dataset"]["event_diagnostics"]["mode"] == "dense"
    assert trial_manifest["dataset"]["sample_weight_diagnostics"]["base_mode"] == "uniform"
    assert trial_manifest["dataset"]["sample_weight_diagnostics"]["concurrency"]["max"] >= 1.0
    assert trial_manifest["meta_sample_weight_diagnostics"]["validation"]["meta_mode"] == "uniform"
    assert trial_manifest["dataset"]["bar_clock_diagnostics"]["serial_correlation"]
    assert trial_manifest["dataset"]["side_data_manifest"]["status"] == "disabled"
