from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel

from .csv_loader import validate_csvs
from .data.bar_sampling import sample_bars
from .data.featureGen import generate_features
from .data.feature_sink import save_features
from .trainer import train
from .backtest.engine import run_backtest
from .utils.logger import init_logging


class Config(BaseModel):
    dataset: str
    trainer: dict
    market_maker: dict
    foundry: dict
    flags: dict


def load_config(path: str) -> Config:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)


def ingest_raw_csvs(cfg: Config):
    raw_dir = Path("data/raw")
    validate_csvs(raw_dir, Path("data/schema.json"), Path("rejects"))


def build_bars_and_features(cfg: Config) -> pd.DataFrame:
    csv_file = next(Path("data/raw").glob("*.csv"))
    df = pd.read_csv(csv_file, header=None, names=["timestamp", "open", "high", "low", "close", "volume"], usecols=range(6))
    bars = sample_bars(df, "tick")
    features = generate_features(bars)
    save_features(cfg.dataset, "tick", features)
    return features


def maybe_train_models(cfg: Config, features: pd.DataFrame) -> Path:
    return train(features, cfg.trainer.get("framework", "pytorch"), Path("artifacts/models"))


def run_backtest_and_sim(cfg: Config, features: pd.DataFrame, model_path: Path):
    if cfg.flags.get("enable_dex_sim"):
        raise NotImplementedError("DEX simulation")
    preds = pd.Series(0, index=features.index)
    run_backtest(features, preds, Path("results") / f"run-{int(datetime.now().timestamp())}")


def summarise_results(cfg: Config):
    summ = Path("results/summary.csv")
    summ.touch(exist_ok=True)
    with summ.open("a") as f:
        f.write(f"{datetime.now().isoformat()},ok\n")


def main():
    import sys
    cfg = load_config(sys.argv[1])
    with init_logging(cfg):
        ingest_raw_csvs(cfg)
        features = build_bars_and_features(cfg)
        model_path = maybe_train_models(cfg, features)
        run_backtest_and_sim(cfg, features, model_path)
        summarise_results(cfg)



if __name__ == "__main__":
    main()

