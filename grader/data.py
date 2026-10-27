"""Verified monthly lake reads and explicit availability-time joins.

Times are UTC and source windows are half-open on bar-open/payment timestamps.
No module in this package imports the editable research library.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RESOLUTIONS = {'1m': 60_000, '5m': 300_000, '1h': 3_600_000}
DATASETS = {'candles_1m': '1m', 'candles_5m': '5m', 'candles_1h': '1h',
            'fundings_1h': '1h', 'mark_price_5m': '5m'}
CANDLE_COLUMNS = ['timestamp_ms', 'timestamp', 'open', 'high', 'low', 'close',
                  'volume', 'quote_asset_volume', 'last_trade_id', 'sample_count',
                  'open_raw', 'high_raw', 'low_raw', 'close_raw', 'market_id', 'symbol']
FUNDING_COLUMNS = ['timestamp_ms', 'timestamp', 'value', 'rate', 'direction', 'market_id', 'symbol']
INTEGER_COLUMNS = {'timestamp_ms', 'last_trade_id', 'sample_count', 'market_id'}


class DataContractError(ValueError):
    """Unknown, corrupted or incomplete evidence; callers must fail closed."""


def utc_ms(value) -> int:
    """Explicit strings/datetimes only; naive calendar strings denote UTC."""
    if value is None or isinstance(value, (int, float, bool, np.number)):
        raise DataContractError('Use explicit UTC dates, not ambiguous numeric dates')
    try:
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
        if pd.isna(stamp) or stamp.value % 1_000_000:
            raise ValueError('Invalid or sub-millisecond timestamp')
        return stamp.value // 1_000_000
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataContractError(f'Invalid UTC timestamp: {value}') from exc


def _iso(value: int | None) -> str | None:
    return pd.Timestamp(value, unit='ms', tz='UTC').isoformat() if value is not None else None


def _json_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataContractError(f'Duplicate manifest key: {key}')
        result[key] = value
    return result


@dataclass(frozen=True)
class LoadedDataset:
    frame: pd.DataFrame
    provenance: dict


@dataclass(frozen=True)
class MarketData:
    datasets: dict[str, LoadedDataset]

    @property
    def frame(self) -> pd.DataFrame:
        return pd.concat([item.frame.assign(market=market) for market, item in self.datasets.items()],
                         ignore_index=True).sort_values(['timestamp_ms', 'market']).reset_index(drop=True)

    @property
    def provenance(self) -> dict:
        return {market: item.provenance for market, item in self.datasets.items()}


def _validate_frame(frame, columns, market, market_id, interval, month):
    if set(frame.columns) != set(columns) or not frame.columns.is_unique:
        raise DataContractError(f'Unknown or missing columns in {market}/{month}')
    strings = {'timestamp', 'symbol', 'direction'}
    for column in set(columns) - strings:
        values = frame[column]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise DataContractError(f'Column {column} must be numeric')
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise DataContractError(f'Nonfinite source values in {column}')
        if column in INTEGER_COLUMNS and not pd.api.types.is_integer_dtype(values):
            raise DataContractError(f'Column {column} must have integer storage')
    if not frame.symbol.eq(market).all() or not frame.market_id.eq(market_id).all():
        raise DataContractError('Market identity mismatch')
    if frame.timestamp_ms.mod(interval).any():
        raise DataContractError('Source timestamps are off the resolution grid')
    clock = pd.to_datetime(frame.timestamp_ms, unit='ms', utc=True)
    stored = pd.to_datetime(frame.timestamp, utc=True, errors='coerce')
    if stored.isna().any() or not stored.eq(clock).all():
        raise DataContractError('timestamp disagrees with timestamp_ms')
    if not clock.dt.strftime('%Y-%m.parquet').eq(month).all():
        raise DataContractError('Timestamp outside its monthly partition')
    if 'direction' in columns:
        if not frame.direction.isin(['long', 'short']).all() or frame.rate.lt(0).any():
            raise DataContractError('Funding requires nonnegative magnitude and explicit payer direction')
    else:
        if (frame[['open', 'high', 'low', 'close']] <= 0).any().any():
            raise DataContractError('Nonpositive prices')
        if (frame[['volume', 'quote_asset_volume', 'sample_count', 'last_trade_id']] < 0).any().any():
            raise DataContractError('Negative volume or counters')
        if (frame.high < frame[['open', 'close', 'low']].max(axis=1)).any() or (
                frame.low > frame[['open', 'close', 'high']].min(axis=1)).any():
            raise DataContractError('Invalid OHLC geometry')


def _gaps(stamps: np.ndarray, start: int, end: int, interval: int, first_known: int) -> list[dict]:
    """Compact missing runs, including requested edges; no synthetic data rows."""
    first_grid = -(-start // interval) * interval
    bounds = np.r_[first_grid - interval, stamps, -(-end // interval) * interval]
    result = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        missing_start, missing_end = int(left + interval), int(right)
        if missing_start < missing_end:
            result.append({'start_ms': missing_start, 'end_ms': missing_end,
                           'count': (missing_end - missing_start) // interval,
                           'kind': 'before_source_start' if missing_end <= first_known else 'missing'})
    return result


class LakeLoader:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)

    def _contained(self, path: Path) -> Path:
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(self.root):
                raise DataContractError('Lake path escapes the allowed root')
            return resolved
        except FileNotFoundError as exc:
            raise DataContractError(f'Missing source path: {path}') from exc

    def load(self, market: str, dataset: str, resolution: str, start, end, *,
             require_complete: bool = True) -> LoadedDataset:
        if not isinstance(market, str) or not re.fullmatch(r'[A-Z][A-Z0-9]{0,15}', market):
            raise DataContractError('Invalid market')
        if dataset not in DATASETS or DATASETS[dataset] != resolution:
            raise DataContractError('Unknown dataset or incompatible resolution')
        begin, finish = utc_ms(start), utc_ms(end)
        if finish <= begin:
            raise DataContractError('Window must be nonempty and half-open')
        manifest_path = self._contained(self.root / market / '_manifest.json')
        manifest_bytes = manifest_path.read_bytes()
        try:
            manifest = json.loads(manifest_bytes, object_pairs_hook=_json_unique)
            market_id = manifest['market_id']
            contract = manifest['datasets'][dataset]
            hashes = contract['partitions']
            interval = RESOLUTIONS[resolution]
            if manifest['symbol'] != market or type(market_id) is not int or market_id < 0:
                raise DataContractError('Invalid manifest market identity')
            if type(contract['expected_interval_ms']) is not int or contract['expected_interval_ms'] != interval:
                raise DataContractError('Manifest resolution mismatch')
            first_known = utc_ms(contract['first'])
            if first_known > utc_ms(contract['last']):
                raise DataContractError('Invalid source bounds')
            if not isinstance(hashes, dict) or not hashes:
                raise DataContractError('Missing partition hashes')
            for name, value in hashes.items():
                if not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])\.parquet', name) or not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
                    raise DataContractError('Invalid manifest partition name or hash')
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DataContractError('Malformed dataset manifest') from exc
        columns = FUNDING_COLUMNS if dataset == 'fundings_1h' else CANDLE_COLUMNS
        frames, partitions = [], []
        first_month = pd.Timestamp(begin, unit='ms', tz='UTC').tz_localize(None).to_period('M')
        last_month = pd.Timestamp(finish - 1, unit='ms', tz='UTC').tz_localize(None).to_period('M')
        for month in pd.period_range(first_month, last_month, freq='M'):
            name = f'{month}.parquet'
            if name not in hashes:
                raise DataContractError(f'Missing manifest partition: {market}/{dataset}/{name}')
            path = self._contained(self.root / market / dataset / name)
            # Deserialize the exact bytes verified, closing the hash/read race.
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != hashes[name]:
                raise DataContractError(f'Partition SHA-256 mismatch: {path}')
            try:
                frame = pd.read_parquet(io.BytesIO(content))
            except Exception as exc:
                raise DataContractError(f'Invalid parquet: {path}') from exc
            _validate_frame(frame, columns, market, market_id, interval, name)
            selected = frame[(frame.timestamp_ms >= begin) & (frame.timestamp_ms < finish)]
            partitions.append({'file': f'{market}/{dataset}/{name}', 'sha256': digest,
                               'rows': len(frame), 'selected_rows': len(selected),
                               'first_ms': int(frame.timestamp_ms.min()) if len(frame) else None,
                               'last_ms': int(frame.timestamp_ms.max()) if len(frame) else None})
            frames.append(frame)
        full = pd.concat(frames, ignore_index=True)[columns]
        duplicate_rows = full[full.timestamp_ms.duplicated(keep=False)]
        if duplicate_rows.groupby('timestamp_ms').nunique(dropna=False).gt(1).any().any():
            raise DataContractError('Conflicting duplicate timestamps')
        full = full.drop_duplicates('timestamp_ms').sort_values('timestamp_ms')
        frame = full[(full.timestamp_ms >= begin) & (full.timestamp_ms < finish)].reset_index(drop=True)
        gaps = _gaps(frame.timestamp_ms.to_numpy(dtype=np.int64), begin, finish, interval, first_known)
        unexpected = [gap for gap in gaps if not (dataset == 'mark_price_5m' and gap['kind'] == 'before_source_start')]
        if require_complete and unexpected:
            raise DataContractError(f'Missing required source observations: {unexpected}')
        frame['available_at_ms'] = frame.timestamp_ms + (0 if dataset == 'fundings_1h' else interval)
        provenance = {'market': market, 'market_id': market_id, 'dataset': dataset, 'resolution': resolution,
                      'window': {'start_ms': begin, 'end_ms': finish},
                      'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                      'partitions': partitions, 'rows': len(frame),
                      'first_ms': int(frame.timestamp_ms.iloc[0]) if len(frame) else None,
                      'last_ms': int(frame.timestamp_ms.iloc[-1]) if len(frame) else None,
                      'first': _iso(int(frame.timestamp_ms.iloc[0])) if len(frame) else None,
                      'last': _iso(int(frame.timestamp_ms.iloc[-1])) if len(frame) else None,
                      'gaps': gaps, 'complete': not gaps, 'unavailable': bool(gaps)}
        return LoadedDataset(frame, provenance)

    def load_markets(self, markets: Iterable[str], dataset: str, resolution: str, start, end,
                     *, require_complete: bool = True) -> MarketData:
        markets = list(markets)
        if not markets or len(set(markets)) != len(markets):
            raise DataContractError('Declare unique, nonempty markets')
        return MarketData({market: self.load(market, dataset, resolution, start, end,
                                            require_complete=require_complete) for market in markets})


def availability_join(decision_times: pd.DatetimeIndex, source: LoadedDataset,
                      columns: Iterable[str]) -> pd.DataFrame:
    """As-of join of actual publications; an overdue observation stays unavailable.

    Callers must load preceding context explicitly. Missing rows/values are never
    carried across a missing scheduled publication or replaced with zero.
    """
    columns = list(columns)
    if not columns or len(set(columns)) != len(columns) or set(columns) - set(source.frame.columns):
        raise DataContractError('Unknown, duplicate or empty join columns')
    if set(columns) & {'decision_time', 'available', 'source_timestamp_ms', 'source_available_at_ms'}:
        raise DataContractError('Reserved join column')
    times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True))
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise DataContractError('Decision times must be valid, unique and increasing')
    if (times.as_unit('ns').asi8 % 1_000_000).any():
        raise DataContractError('Decision times must use millisecond precision')
    clock = times.as_unit('ns').asi8 // 1_000_000
    interval = RESOLUTIONS[source.provenance['resolution']]
    right = source.frame[['timestamp_ms', 'available_at_ms', *[c for c in columns if c not in {'timestamp_ms', 'available_at_ms'}]]].copy()
    right = right.rename(columns={'timestamp_ms': 'source_timestamp_ms', 'available_at_ms': 'source_available_at_ms'})
    aliases = {'timestamp_ms': 'source_timestamp_ms', 'available_at_ms': 'source_available_at_ms'}
    selected = [aliases.get(column, column) for column in columns]
    left = pd.DataFrame({'decision_at_ms': clock})
    joined = pd.merge_asof(left, right.sort_values('source_available_at_ms'),
                           left_on='decision_at_ms', right_on='source_available_at_ms',
                           direction='backward', tolerance=interval - 1)
    result = joined[['source_timestamp_ms', 'source_available_at_ms', *[c for c in selected if c not in {'source_timestamp_ms', 'source_available_at_ms'}]]].copy()
    result.insert(0, 'decision_time', times)
    result['available'] = joined.source_timestamp_ms.notna()
    return result
