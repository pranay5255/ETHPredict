"""Strict file contracts shared by the trusted runner and isolated child."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re
import stat

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import DataContractError, LoadedDataset, MarketData, RESOLUTIONS
from .env import ReplayData, ReplayEnv, TorchReplayBatch
from .feature_contract import FEATURE_FAMILY_COLUMNS


@dataclass(frozen=True)
class Declaration:
    name: str
    markets: tuple[str, ...]
    resolution: str
    max_horizon_bars: int
    lookback_bars: int
    budget_tier: str
    engineered_features: tuple[str, ...] = ()
    pretrained_models: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value):
        fields = {'name', 'markets', 'resolution', 'max_horizon_bars', 'lookback_bars',
                  'budget_tier', 'engineered_features', 'pretrained_models'}
        if not isinstance(value, dict) or set(value) != fields:
            raise DataContractError('Unknown or missing strategy declaration fields')
        if not isinstance(value['name'], str) or not re.fullmatch(r'[A-Za-z0-9_ .-]{1,128}', value['name']):
            raise DataContractError('Invalid strategy name')
        markets = value['markets']
        if not isinstance(markets, list) or not markets or len(set(markets)) != len(markets) or any(
                not isinstance(m, str) or not re.fullmatch(r'[A-Z][A-Z0-9]{0,15}', m) for m in markets):
            raise DataContractError('Declare a nonempty unique market list')
        if value['resolution'] not in RESOLUTIONS or value['budget_tier'] not in {'fast', 'long'}:
            raise DataContractError('Unknown clock or budget tier')
        for name in ['max_horizon_bars', 'lookback_bars']:
            if type(value[name]) is not int or value[name] < 1:
                raise DataContractError(f'{name} must be a positive integer')
        for name in ['engineered_features', 'pretrained_models']:
            values = value[name]
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(set(values)) != len(values):
                raise DataContractError(f'Invalid {name}')
        if set(value['engineered_features']) - set(FEATURE_FAMILY_COLUMNS):
            raise DataContractError('Unknown engineered feature family')
        return cls(**{**value, 'markets': tuple(markets),
                      'engineered_features': tuple(value['engineered_features']),
                      'pretrained_models': tuple(value['pretrained_models'])})


@dataclass(frozen=True)
class ReplayFactory:
    """Only the training arrays supplied to this fit exist in this factory."""
    data: ReplayData
    seed: int

    def __call__(self, *, window=None, lookback=1, seed=None):
        return ReplayEnv(self.data, window=window, lookback=lookback,
                         seed=self.seed if seed is None else seed)

    def batch(self, batch_size, episode_steps, *, lookback=1, device='cuda', seed=None):
        return TorchReplayBatch(self.data, batch_size, episode_steps, lookback=lookback,
                                device=device, seed=self.seed if seed is None else seed)


@dataclass(frozen=True)
class FitContext:
    seed: int
    device: str
    deadline: float                 # time.monotonic(), all calls share this deadline
    scratch_dir: Path
    replay_env: ReplayFactory | None
    dev_lake: Path | None           # grader-produced, availability-filtered view
    train_anchor_end_ms: int | None


def write_market_data(data: MarketData, directory: Path):
    directory.mkdir(parents=True)
    specification = {}
    for market, item in data.datasets.items():
        if not re.fullmatch(r'[A-Z][A-Z0-9]{0,15}', market):
            raise DataContractError('Invalid market in input')
        item.frame.to_parquet(directory / f'{market}.parquet', index=False)
        specification[market] = {'file': f'{market}.parquet', 'provenance': item.provenance}
    (directory / 'market_data.json').write_text(json.dumps(specification, allow_nan=False))


def read_market_data(directory: Path) -> MarketData:
    specification = json.loads((directory / 'market_data.json').read_text())
    return MarketData({market: LoadedDataset(pd.read_parquet(directory / item['file']), item['provenance'])
                       for market, item in specification.items()})


def write_replay(data: ReplayData, path: Path):
    arrays = {name: value for name, value in vars(data).items() if isinstance(value, np.ndarray)}
    np.savez(path, **arrays)
    metadata = {name: value for name, value in vars(data).items() if name not in arrays}
    path.with_suffix('.json').write_text(json.dumps(metadata, allow_nan=False))


def read_replay(path: Path) -> ReplayData:
    metadata = json.loads(path.with_suffix('.json').read_text())
    metadata['markets'] = tuple(metadata['markets'])
    with np.load(path, allow_pickle=False) as content:
        arrays = {name: content[name] for name in content.files}
    for value in arrays.values():
        value.flags.writeable = False
    return ReplayData(**metadata, **arrays)


def regular_bytes(path: Path, *, maximum: int) -> bytes:
    """Never follow a candidate symlink or deserialize a device/FIFO/hard link."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1 or status.st_size > maximum:
            raise DataContractError('Output must be a bounded regular file with one link')
        with os.fdopen(descriptor, 'rb', closefd=False) as source:
            content = source.read(maximum + 1)
        if len(content) > maximum:
            raise DataContractError('Output exceeded its byte limit')
        return content
    finally:
        os.close(descriptor)


def check_regular_file(path: Path, *, maximum: int):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1 or not 0 < status.st_size <= maximum:
            raise DataContractError('State must be a bounded regular file with one link')
    finally:
        os.close(descriptor)


def read_positions(path: Path, declaration: Declaration, decision_times: pd.DatetimeIndex) -> pd.DataFrame:
    times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True))
    expected_rows = len(times) * len(declaration.markets)
    content = regular_bytes(path, maximum=max(1_048_576, expected_rows * 128))
    parquet = pq.ParquetFile(io.BytesIO(content))
    schema = parquet.schema_arrow
    if parquet.metadata.num_rows != expected_rows or schema.names != ['timestamp', 'market', 'target_position']:
        raise DataContractError('Positions require the exact schema and complete market/time grid')
    stamp, market, target = [field.type for field in schema]
    if not pa.types.is_timestamp(stamp) or stamp.tz != 'UTC' or not (
            pa.types.is_string(market) or pa.types.is_large_string(market)) or not pa.types.is_floating(target):
        raise DataContractError('Positions require UTC timestamps, string markets and floating targets')
    frame = parquet.read().to_pandas()
    if frame.isna().any().any() or not np.isfinite(frame.target_position).all() or frame.target_position.abs().gt(1).any():
        raise DataContractError('Nonfinite or out-of-cap positions')
    index = pd.MultiIndex.from_product([times, sorted(declaration.markets)], names=['timestamp', 'market'])
    actual = frame.set_index(['timestamp', 'market']).sort_index()
    if not actual.index.is_unique or not actual.index.equals(index):
        raise DataContractError('Positions do not cover every declared market on the decision grid')
    return actual.reset_index()
