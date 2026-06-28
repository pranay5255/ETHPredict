"""Compatibility entrypoints for the lightweight Lighter-only workflow.

The previous all-in-one multi-source pipeline is archived at
``archive/legacy_data_sources/code/src_main_multisource.py``. Active pipeline
orchestration lives in the top-level ``runner.py`` and the Lighter-only feature
code lives in ``src.data.features_all``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from pydantic import BaseModel as PydanticBaseModel

from src.csv_loader import validate_csvs


class Config(PydanticBaseModel):
    dataset: str
    trainer: Dict[str, Any]
    market_maker: Dict[str, Any]
    foundry: Dict[str, Any]
    flags: Dict[str, Any]


def load_config(path: str) -> Config:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)


def ingest_raw_csvs(
    cfg: Config,
    raw_dir: Path = Path("data/raw"),
    schema_path: Path = Path("data/schema.json"),
    rejects_dir: Path = Path("rejects"),
):
    return validate_csvs(raw_dir, schema_path, rejects_dir)


def run_backtest_and_sim(cfg: Config, features: pd.DataFrame, model_path: Optional[Path]):
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.main <config.yml>")
        return
    cfg = load_config(sys.argv[1])
    ingest_raw_csvs(cfg)


if __name__ == "__main__":
    main()
