"""Availability-time historical side features with explicit coverage gates.

Sources must be historical time series, not overwritten latest snapshots. Each
source declares when its values became available; event time alone is not an
implicit publication-time guarantee.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_INPUTS = {
    'funding': ['funding_rate'],
    'mark_basis': ['mark_price'],
    'open_interest': ['open_interest'],
    'exchange_volume': ['exchange_volume'],
    'order_book': ['bid_price', 'ask_price', 'bid_depth', 'ask_depth'],
    'trade_flow': ['buy_volume', 'sell_volume'],
}
GROUP_DERIVED = {'mark_basis': ['mark_basis'], 'order_book': ['spread_bps', 'book_imbalance'],
                 'trade_flow': ['trade_flow_imbalance']}


class DataCoverageError(ValueError):
    def __init__(self, message, report):
        self.report = report
        super().__init__(f'{message}: {json.dumps(report, default=str, sort_keys=True)}')


def enabled_groups(config):
    groups = (config or {}).get('groups', {})
    if not isinstance(groups, dict):
        raise ValueError('data.side_data.groups must map group names to source configurations')
    if any(not isinstance(value, dict) for value in groups.values()):
        raise ValueError('Each side-data group must be a source configuration mapping')
    enabled = {name: value for name, value in sorted(groups.items()) if value.get('enabled', True)}
    unknown = set(enabled) - set(GROUP_INPUTS)
    if unknown:
        raise ValueError(f'Unknown side-data groups: {sorted(unknown)}')
    return enabled


def side_feature_columns(config):
    return [column for group in enabled_groups(config)
            for column in GROUP_INPUTS[group] + GROUP_DERIVED.get(group, []) + [f'{group}_missing']]


def source_paths(data_dir, config):
    paths = []
    for name, source in enabled_groups(config).items():
        configured = source.get('paths', [])
        if not configured or isinstance(configured, str):
            raise ValueError(f'{name}: paths must be a nonempty explicit list of historical files')
        for value in configured:
            path = Path(value)
            if not path.is_absolute():
                path = Path(data_dir) / path
            if not path.is_file():
                raise ValueError(f'{name}: historical source file missing: {path}')
            paths.append((name, path))
    return paths


def _time(values, unit):
    if pd.api.types.is_numeric_dtype(values) and unit is None:
        raise ValueError('Numeric side-data timestamps require an explicit timestamp unit')
    return pd.to_datetime(values, unit=unit, utc=True, errors='coerce').dt.tz_localize(None).astype('datetime64[ns]')


def join_configured_side_data(features, price_data, config, data_dir):
    groups = enabled_groups(config)
    if not groups:
        return features, {'enabled_groups': [], 'coverage': {}, 'status': 'disabled', 'source_files': []}
    output = features.copy()
    # A completed candle's OHLCV is available at close, not its opening timestamp.
    close_time = pd.to_numeric(price_data['close_time'], errors='raise')
    anchor = pd.DataFrame({'_row': np.arange(len(output)), '_asof': pd.to_datetime(close_time.to_numpy(), unit='ms', utc=True).tz_localize(None)})
    anchor['_asof'] = anchor['_asof'].astype('datetime64[ns]')
    reports, files = {}, []
    for group, source in groups.items():
        if source.get('historical') is not True:
            raise ValueError(f'{group}: historical: true is required; latest snapshots are not historical inputs')
        if 'available_at_column' not in source and 'availability_delay' not in source:
            raise ValueError(f'{group}: declare available_at_column or availability_delay')
        fill = source.get('fill_policy', 'reject')
        if fill not in {'reject', 'zero_with_indicator'}:
            raise ValueError(f'{group}: unsupported fill_policy {fill}; future or unbounded filling is prohibited')
        tolerance = pd.Timedelta(source.get('tolerance', '0s'))
        if tolerance < pd.Timedelta(0):
            raise ValueError(f'{group}: tolerance must be nonnegative')
        frames = []
        for name, path in source_paths(data_dir, {'groups': {group: source}}):
            if 'latest' in path.stem.lower():
                raise ValueError(f'{group}: overwritten latest snapshot cannot be a frozen historical source: {path}')
            frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
            event_column = source.get('timestamp_column', 'timestamp')
            frame['_event'] = _time(frame[event_column], source.get('timestamp_unit'))
            available_column = source.get('available_at_column')
            if available_column:
                frame['_available'] = _time(frame[available_column], source.get('available_at_unit', source.get('timestamp_unit')))
            else:
                delay = pd.Timedelta(source['availability_delay'])
                if delay < pd.Timedelta(0):
                    raise ValueError(f'{group}: availability_delay cannot be negative')
                frame['_available'] = frame['_event'] + delay
            if frame[['_event', '_available']].isna().any().any() or (frame['_available'] < frame['_event']).any():
                raise ValueError(f'{group}: invalid availability times or publication before event')
            mapping = source.get('columns', {column: column for column in GROUP_INPUTS[group]})
            frame = frame.rename(columns=mapping)
            missing = set(GROUP_INPUTS[group]) - set(frame)
            if missing:
                raise ValueError(f'{group}: missing canonical historical fields: {sorted(missing)}')
            frame = frame[['_event', '_available', *GROUP_INPUTS[group]]].copy()
            for column in GROUP_INPUTS[group]:
                frame[column] = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
            frames.append(frame)
            files.append({'group': group, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
        side = pd.concat(frames, ignore_index=True)
        source_monotonic = bool(side['_available'].is_monotonic_increasing)
        duplicates = int(side['_available'].duplicated().sum())
        if side.groupby('_available')[['_event', *GROUP_INPUTS[group]]].nunique(dropna=False).gt(1).any().any():
            raise ValueError(f'{group}: conflicting duplicate availability timestamps')
        side = side.drop_duplicates('_available').sort_values('_available')
        if len(side) < max(2, int(source.get('min_source_rows', 2))):
            raise DataCoverageError(f'{group}: insufficient historical rows', {'source_rows': len(side), 'required': max(2, int(source.get('min_source_rows', 2)))})
        joined = pd.merge_asof(anchor.sort_values('_asof'), side, left_on='_asof', right_on='_available',
                               direction='backward', tolerance=tolerance).sort_values('_row')
        for column in GROUP_INPUTS[group]:
            output[column] = joined[column].to_numpy(dtype=float)
        if group == 'mark_basis':
            output['mark_basis'] = output['mark_price'] / output['close'] - 1.
        if group == 'order_book':
            valid_book = (output['bid_price'] > 0) & (output['ask_price'] >= output['bid_price']) & (output['bid_depth'] >= 0) & (output['ask_depth'] >= 0)
            midpoint = (output['ask_price'] + output['bid_price']) / 2
            output['spread_bps'] = ((output['ask_price'] - output['bid_price']) / midpoint * 10_000).where(valid_book)
            total = output['bid_depth'] + output['ask_depth']
            output['book_imbalance'] = ((output['bid_depth'] - output['ask_depth']) / total.replace(0, np.nan)).where(valid_book)
        if group == 'trade_flow':
            total = output['buy_volume'] + output['sell_volume']
            valid_flow = (output['buy_volume'] >= 0) & (output['sell_volume'] >= 0)
            output['trade_flow_imbalance'] = ((output['buy_volume'] - output['sell_volume']) / total.replace(0, np.nan)).where(valid_flow)
            output.loc[valid_flow & total.eq(0), 'trade_flow_imbalance'] = 0.
        columns = GROUP_INPUTS[group] + GROUP_DERIVED.get(group, [])
        missing_mask = ~np.isfinite(output[columns]).all(axis=1)
        if group in {'mark_basis', 'open_interest', 'exchange_volume'}:
            missing_mask |= output[GROUP_INPUTS[group][0]] < (1e-15 if group == 'mark_basis' else 0.)
        observed = ~missing_mask
        availability = joined['_available']
        if (availability.dropna().to_numpy() > joined.loc[availability.notna(), '_asof'].to_numpy()).any():
            raise AssertionError('Side-data join used a future observation')
        report = {'rows': len(output), 'source_rows': len(side), 'source_monotonic': source_monotonic,
                  'duplicate_rows_removed': duplicates, 'coverage': float(observed.mean()),
                  'missing_rows': int(missing_mask.sum()), 'tolerance_seconds': tolerance.total_seconds(),
                  'fill_policy': fill, 'availability_contract': {key: source[key] for key in
                  ['timestamp_column', 'timestamp_unit', 'available_at_column', 'available_at_unit', 'availability_delay'] if key in source},
                  'earliest_available': str(side['_available'].min()), 'latest_available': str(side['_available'].max()),
                  'max_join_age_seconds': float((joined['_asof'] - availability).dt.total_seconds().max()) if availability.notna().any() else None,
                  'future_matches': 0}
        minimum = float(source.get('min_coverage', 1.))
        if not 0 <= minimum <= 1:
            raise ValueError(f'{group}: min_coverage must be between zero and one')
        if report['coverage'] < minimum or (fill == 'reject' and missing_mask.any()):
            raise DataCoverageError(f'{group}: historical coverage gate failed', report)
        output[f'{group}_missing'] = missing_mask.astype(float)
        output.loc[missing_mask, columns] = 0.
        reports[group] = report
    return output, {'status': 'validated', 'enabled_groups': list(groups), 'coverage': reports,
                    'source_files': files, 'join_clock': 'completed_bar_close', 'future_matches': 0}


def validate_split_coverage(dataset, splits, *, include_final=True):
    """Gate the actual feature rows used by each model partition before fitting."""
    config = dataset.get('coverage_config', {})
    side_groups = enabled_groups(config.get('side_data', {}))
    bars = config.get('bars', {})
    default_rows = 2 if bars.get('type', 'time') != 'time' else 1
    min_rows = bars.get('min_rows_per_split', {})
    parts = [('development', splits['development']), ('test', splits['test'])] if include_final else []
    for fold in splits['folds']:
        parts.extend([(f'fold_{fold["fold"]}_train', fold['train']),
                      (f'fold_{fold["fold"]}_validation', fold['validation'])])
    reports = {}
    feature_frame = dataset.get('feature_frame')
    for name, indices in parts:
        role = 'train' if name.endswith('train') or name == 'development' else 'test' if name == 'test' else 'validation'
        minimum = int(min_rows.get(role, default_rows))
        report = {'samples': len(indices), 'minimum_samples': minimum, 'side_data': {}}
        if len(indices) < minimum:
            raise DataCoverageError(f'{name}: insufficient split coverage', report)
        source_indices = dataset['samples'].iloc[indices]['sample_index'].to_numpy(dtype=int)
        used = np.zeros(len(dataset['price_path']) + 1, dtype=int)
        np.add.at(used, source_indices - dataset['sequence_length'], 1)
        np.add.at(used, source_indices, -1)
        used = np.flatnonzero(np.cumsum(used[:-1]) > 0)
        if 'source_timestamps' in dataset:
            time = pd.DatetimeIndex(dataset['source_timestamps']).astype('datetime64[ns]').asi8
            actual, expected = 0, 0
            interval = int(config['source_interval_ms']) * 1_000_000
            for run in np.split(used, np.flatnonzero(np.diff(used) > 1) + 1):
                if not len(run):
                    continue
                left = int(dataset['price_path'].iloc[run[0]]['open_time']) * 1_000_000
                right = int(dataset['price_path'].iloc[run[-1]]['close_time']) * 1_000_000
                actual += int(np.searchsorted(time, right, side='right') - np.searchsorted(time, left))
                expected += (right - left + 1_000_000) // interval
            report['source_interval_coverage'] = actual / expected if expected else 0.
            report['missing_source_intervals'] = expected - actual
            allowed = float(bars.get('max_missing_fraction', 0. if bars.get('type', 'time') != 'time' else 1.))
            if expected == 0 or (expected - actual) / expected > allowed:
                raise DataCoverageError(f'{name}: source interval split coverage gate failed', report)
        if feature_frame is not None and side_groups:
            # Union input windows; source-bar and event-sample clocks differ.
            report['input_bars'] = len(used)
            for group, source in side_groups.items():
                observed = 1. - float(feature_frame.iloc[used][f'{group}_missing'].mean()) if len(used) else 0.
                report['side_data'][group] = observed
                minimum_coverage = float(source.get('min_split_coverage', source.get('min_coverage', 1.)))
                if not 0 <= minimum_coverage <= 1:
                    raise ValueError(f'{group}: min_split_coverage must be between zero and one')
                if observed < minimum_coverage:
                    raise DataCoverageError(f'{name}: {group} split coverage gate failed', report)
        reports[name] = report
    return reports
