from __future__ import annotations
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models.simple import LSTMModel, MetaMLP, ConfidenceGRU


def _train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer):
    model.train()
    loss_total = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return loss_total / len(loader)


def train_pytorch(df: pd.DataFrame, out_dir: Path, model_cls= LSTMModel):
    features = torch.tensor(df[["return", "rolling_vol"]].values, dtype=torch.float32)
    targets = torch.tensor(df["return"].shift(-1).fillna(0).values, dtype=torch.float32)

    dataset = TensorDataset(features.unsqueeze(1), targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = model_cls(features.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(3):
        _train_epoch(model, loader, criterion, optimizer)

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = out_dir / f"model_{ts}.pt"
    torch.save(model.state_dict(), path)
    return path


def train_ray(df: pd.DataFrame, out_dir: Path):
    import ray

    @ray.remote
    def _remote_train(data: pd.DataFrame) -> str:
        return str(train_pytorch(data, out_dir))

    ray.init(ignore_reinit_error=True)
    ref = _remote_train.remote(df)
    result = ray.get(ref)
    ray.shutdown()
    return Path(result)


FRAMEWORKS = {
    "pytorch": train_pytorch,
    "ray": train_ray,
}


def train(df: pd.DataFrame, framework: str, out_dir: Path) -> Path:
    if framework not in FRAMEWORKS:
        raise NotImplementedError(framework)
    trainer_fn = FRAMEWORKS[framework]
    return trainer_fn(df, out_dir)


