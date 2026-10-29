"""Numeric policy-input protocol for trusted, isolated permutation checks."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import ndtri

from .contracts import regular_bytes
from .data import DataContractError, RESOLUTIONS, _json_unique


NOISE_NAMES = ('diagnostic_noise_0', 'diagnostic_noise_1')
MAX_FEATURE_BYTES = 4 * 1000**3


def feature_noise(timestamps_ms, *, seed, market):
    """Stateless float32 Gaussian controls, stable under append/truncation.

    SplitMix64 mixes the explicit UTC millisecond clock with grader seed,
    market and channel; inverse-normal CDF converts open-interval uniforms.
    These generated controls are not substitutes for missing observations.
    """
    clock = np.asarray(timestamps_ms)
    if clock.ndim != 1 or not np.issubdtype(clock.dtype, np.integer) or type(seed) is not int:
        raise DataContractError('Noise controls require integer UTC milliseconds and seed')
    result = []
    for name in NOISE_NAMES:
        key = int.from_bytes(hashlib.sha256(f'{seed}:{market}:{name}'.encode()).digest()[:8], 'little')
        with np.errstate(over='ignore'):
            value = clock.astype(np.uint64) + np.uint64(key) + np.uint64(0x9e3779b97f4a7c15)
            value = (value ^ (value >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
            value = (value ^ (value >> 27)) * np.uint64(0x94d049bb133111eb)
            value ^= value >> 31
        uniform = ((value >> 11).astype(np.float64) + 1.) / (2**53 + 2.)
        result.append(ndtri(uniform).astype(np.float32))
    return np.stack(result, axis=1)


@dataclass(frozen=True)
class FeaturePacket:
    frame: pd.DataFrame
    families: dict[str, list[str]]
    noise: dict[str, list[str]]
    noise_lags: dict[str, int]

    @property
    def groups(self):
        return {**self.families, **self.noise}


def write_packet(packet: FeaturePacket, directory):
    if not isinstance(packet, FeaturePacket):
        raise TypeError('feature_packet must return grader.feature_packets.FeaturePacket')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    packet.frame.to_parquet(directory/'features.parquet', index=False)
    metadata = {'version':1, 'families':packet.families, 'noise':packet.noise, 'noise_lags':packet.noise_lags}
    (directory/'features.json').write_text(json.dumps(metadata, allow_nan=False))


def read_packet(directory, declaration, decision_times, *, seed, verify_noise=True):
    directory = Path(directory)
    metadata = json.loads(regular_bytes(directory/'features.json', maximum=4*1024**2), object_pairs_hook=_json_unique)
    if set(metadata) != {'version','families','noise','noise_lags'} or metadata['version'] != 1:
        raise DataContractError('Unknown feature-packet fields/version')
    families, noise, lags = metadata['families'], metadata['noise'], metadata['noise_lags']
    if not isinstance(families,dict) or set(families) != set(declaration.engineered_features) or not isinstance(noise,dict) or set(noise) != set(NOISE_NAMES):
        raise DataContractError('Feature families/noise controls must match the pre-run declaration')
    groups = {**families, **noise}
    columns = []
    for names in groups.values():
        if not isinstance(names,list) or not names or any(not isinstance(n,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.:-]{0,127}',n) for n in names):
            raise DataContractError('Invalid policy-input column group')
        columns.extend(names)
    if len(columns) != len(set(columns)) or set(columns) & {'timestamp','market','target_position'} or len(columns)>32768:
        raise DataContractError('Policy inputs must have unique, partitioned numeric columns')
    noise_columns = {column for names in noise.values() for column in names}
    if not isinstance(lags,dict) or set(lags) != noise_columns or any(
            type(lag) is not int or not 0 <= lag <= declaration.lookback_bars for lag in lags.values()):
        raise DataContractError('Every noise input needs a causal lag within declared lookback')
    times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True)).as_unit('ns')
    rows = len(times)*len(declaration.markets)
    if rows*len(columns)*8 > MAX_FEATURE_BYTES:
        raise DataContractError('Decoded feature packet exceeds 4 GB')
    raw = regular_bytes(directory/'features.parquet', maximum=MAX_FEATURE_BYTES)
    parquet = pq.ParquetFile(io.BytesIO(raw))
    schema = parquet.schema_arrow
    if parquet.metadata.num_rows != rows or len(schema.names) != len(columns)+2 or set(schema.names) != {'timestamp','market',*columns}:
        raise DataContractError('Feature packet does not cover the full declared grid/schema')
    for field in schema:
        if field.name == 'timestamp':
            valid = pa.types.is_timestamp(field.type) and field.type.tz == 'UTC'
        elif field.name == 'market':
            valid = pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
        else:
            valid = pa.types.is_floating(field.type)
        if not valid:
            raise DataContractError('Feature packet requires UTC, string market and floating inputs')
    frame = parquet.read().to_pandas()
    # Normalize storage units before index comparison. Pandas/Arrow can retain
    # either ms or us, and mixed-unit MultiIndex equality depends on its cache.
    # Nanoseconds preserve every supplied timestamp; no rounding is permitted.
    frame['timestamp'] = frame.timestamp.dt.as_unit('ns')
    expected = pd.MultiIndex.from_product([times,sorted(declaration.markets)], names=['timestamp','market'])
    indexed = frame.set_index(['timestamp','market']).sort_index()
    if not indexed.index.is_unique or not indexed.index.equals(expected) or not np.isfinite(indexed.to_numpy(dtype=float)).all():
        raise DataContractError('Nonfinite, duplicate or missing policy inputs')
    frame = indexed.reset_index()
    if verify_noise:
        interval = RESOLUTIONS[declaration.resolution]
        for market in declaration.markets:
            mask = frame.market.eq(market)
            clock = pd.DatetimeIndex(frame.loc[mask,'timestamp']).as_unit('ms').asi8
            for channel, name in enumerate(NOISE_NAMES):
                for column in noise[name]:
                    expected_noise = feature_noise(clock-lags[column]*interval,seed=seed,market=market)[:,channel]
                    if not np.array_equal(frame.loc[mask,column].to_numpy(dtype=float),expected_noise.astype(float)):
                        raise DataContractError('Policy noise differs from the frozen grader controls')
    return FeaturePacket(frame, families, noise, lags)


def same_packet(left, right, *, cutoff=None, tolerance=1e-9):
    if left is None or right is None:
        return left is right
    if (left.families,left.noise,left.noise_lags) != (right.families,right.noise,right.noise_lags):
        return False
    a = left.frame if cutoff is None else left.frame[left.frame.timestamp <= cutoff]
    b = right.frame
    identity = ['timestamp','market']
    a,b = a.set_index(identity).sort_index(),b.set_index(identity).sort_index()
    return list(a)==list(b) and a.index.equals(b.index) and np.allclose(a,b,rtol=0,atol=tolerance)
