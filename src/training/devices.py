from __future__ import annotations

from typing import Optional, Union

import torch


def resolve_training_device(
    device: Optional[Union[str, torch.device]] = None,
    *,
    allow_cpu: bool = False,
) -> torch.device:
    """Resolve the neural training device for experiment runs.

    Neural experiments are GPU-first by default. Passing ``allow_cpu=True`` is
    the explicit escape hatch for CPU smoke tests or local development.
    """
    requested = torch.device(device or "cuda:0")

    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {requested}, but torch.cuda.is_available() is false. "
                "Install the uv CUDA Torch environment or pass --allow-cpu for CPU-only smoke runs."
            )
        if requested.index is not None and requested.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {requested}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        return requested

    if requested.type == "cpu" and not allow_cpu:
        raise RuntimeError("CPU neural training requires --allow-cpu.")

    return requested


def assert_cuda_environment(expected_name: str = "RTX 4090") -> str:
    """Assert that uv-managed Torch can see the target CUDA device."""
    device = resolve_training_device("cuda:0")
    name = torch.cuda.get_device_name(device)
    if expected_name not in name:
        raise RuntimeError(f"Expected CUDA device name to include {expected_name!r}, got {name!r}.")
    return name
