from pathlib import Path

import numpy as np
import torch

from src.experiments.lighter_compare import run_arima_sarimax_smoke, run_experiment
from src.training.devices import assert_cuda_environment, resolve_training_device
from src.training.trainer import hierarchical_training_pipeline


def test_uv_torch_sees_cuda_0_rtx_4090():
    name = assert_cuda_environment("RTX 4090")

    assert torch.cuda.is_available()
    assert "RTX 4090" in name


def test_one_batch_gpu_training_for_three_model_stack():
    device = resolve_training_device("cuda:0")
    X = torch.randn(8, 4, 5)
    y_ret = torch.randn(8, 1)
    y_dir = torch.tensor([1, 0, -1, 1, 0, -1, 1, 0])
    sample_weights = torch.ones(8)

    results = hierarchical_training_pipeline(
        X,
        y_ret,
        y_dir,
        sample_weights,
        input_size=5,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        batch_size=8,
        num_epochs=1,
        device=device,
    )

    assert next(results["level_0"]["model"].parameters()).is_cuda
    assert next(results["level_1"]["model"].parameters()).is_cuda
    assert next(results["level_2"]["model"].parameters()).is_cuda


def test_cpu_arima_sarimax_smoke():
    values = np.sin(np.linspace(0, 4, 40)) * 0.01

    results = run_arima_sarimax_smoke(values)

    assert set(results) == {"arima", "sarimax"}
    assert np.isfinite(results["arima"]["forecast"])
    assert np.isfinite(results["sarimax"]["forecast"])


def test_lighter_compare_smoke_end_to_end():
    result = run_experiment(Path("configs/lighter_experiments.yml"), smoke=True, device="cuda:0")

    assert result["device"] == "cuda:0"
    assert result["stack_smoke"]["models"] == ["PriceLSTM", "MetaMLP", "ConfidenceGRU"]
    assert set(result["targets"]) == {"triple_barrier", "next_hour_return"}
    for target_result in result["targets"].values():
        assert target_result["samples"] > 0
        assert len(target_result["neural_trials"]) == 1
        assert set(target_result["baselines"]) == {"arima", "sarimax"}
