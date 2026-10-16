"""Exact inclusive label-interval purging on the source clock.

Grouped scenarios are independently fitted held-out group combinations. They
are not advertised as stitched CPCV backtest paths.
"""
from itertools import combinations
from math import comb
from typing import Any, Mapping

import numpy as np
import pandas as pd


def interval_coordinates(samples):
    names = ['label_span_start_timestamp', 'label_span_end_timestamp']
    if all(name in samples for name in names):
        arrays = [pd.to_datetime(samples[name], utc=True, errors='coerce') for name in names]
        if any(array.isna().any() for array in arrays):
            raise ValueError('Label spans require valid timestamps')
        starts, ends = [array.astype('datetime64[ns, UTC]').astype('int64').to_numpy() for array in arrays]
    else:
        starts, ends = [pd.to_numeric(samples[name], errors='raise').to_numpy() for name in
                        ['label_span_start_idx', 'label_span_end_idx']]
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or (ends < starts).any():
        raise ValueError('Invalid label intervals')
    return starts, ends


def overlaps(starts, ends, other_starts, other_ends):
    """O((n + m) log m), without materializing an n-by-m overlap matrix."""
    if not len(other_starts):
        return np.zeros(len(starts), dtype=bool)
    order = np.argsort(other_starts)
    left = np.asarray(other_starts)[order]
    right = np.maximum.accumulate(np.asarray(other_ends)[order])
    index = np.searchsorted(left, ends, side='right') - 1
    return (index >= 0) & (right[np.maximum(index, 0)] >= starts)


def embargo_width(samples, evaluation, config, *, max_horizon=1, sequence_length=1, train=None):
    mode = config.get('embargo_mode', 'fixed_bars')
    if mode == 'fixed_bars':
        width = int(config.get('embargo_bars', 0))
    elif mode == 'max_horizon':
        width = int(max_horizon)
    elif mode == 'sequence_plus_horizon':
        width = int(sequence_length + max_horizon)
    elif mode == 'realized_t1':
        part = samples.iloc[evaluation]
        width = int((part.label_span_end_idx - part.label_span_start_idx).max()) if len(part) else 0
    elif mode == 'volatility_adaptive':
        # Only the proposed training population determines adaptive width.
        part = samples.iloc[train] if train is not None else samples.iloc[:0]
        volatility = pd.to_numeric(part.get('realized_vol', pd.Series(dtype=float)), errors='coerce').dropna()
        reference = float(volatility.median()) if len(volatility) else 0.
        ratio = float(volatility.quantile(.95) / reference) if reference > 0 else 1.
        width = int(np.ceil(max_horizon * np.clip(ratio, 1., 10.)))
    else:
        raise ValueError(f'Unknown embargo_mode: {mode}')
    if width < 0:
        raise ValueError('Embargo width must be nonnegative')
    return width


def purge_training_spans(samples, train, evaluation, config=None, *, clock=None, max_horizon=1, sequence_length=1):
    train, evaluation = np.asarray(train, dtype=int), np.asarray(evaluation, dtype=int)
    if any((indices < 0).any() or (indices >= len(samples)).any() for indices in (train, evaluation)):
        raise ValueError('Split row IDs are outside the sample population')
    starts, ends = interval_coordinates(samples)
    config = config or {}
    width = embargo_width(samples, evaluation, config, max_horizon=max_horizon,
                          sequence_length=sequence_length, train=train)
    conflict = overlaps(starts[train], ends[train], starts[evaluation], ends[evaluation])
    embargo_end = ends[evaluation].copy()
    if width:
        if clock is not None:
            time = pd.DatetimeIndex(pd.to_datetime(clock, utc=True)).astype('datetime64[ns, UTC]').asi8
            source_end = samples.iloc[evaluation].label_span_end_idx.to_numpy(dtype=int)
            position = source_end + width
            embargo_end = time[np.minimum(position, len(time) - 1)]
            # Extend past the final observed bar so later intervals cannot slip through.
            embargo_end = np.where(position >= len(time), np.iinfo(np.int64).max, embargo_end)
        elif 'label_span_start_timestamp' in samples:
            raise ValueError('Timestamp embargo requires the source-bar clock')
        else:
            embargo_end = ends[evaluation] + width
    embargoed = ~conflict & overlaps(starts[train], ends[train], ends[evaluation], embargo_end) if width else np.zeros(len(train), dtype=bool)
    retained = train[~(conflict | embargoed)]
    remaining = overlaps(starts[retained], ends[retained], starts[evaluation], ends[evaluation])
    if remaining.any():
        raise AssertionError('Exact label-span purge left an overlap')
    return retained, {'purged_indices': train[conflict].tolist(),
                      'embargoed_indices': train[embargoed].tolist(), 'embargo_bars': width,
                      'overlap_count_after': int(remaining.sum()), 'fit_indices': retained.tolist(),
                      'evaluation_indices': evaluation.tolist()}


def exact_validation_splits(dataset: Mapping[str, Any], config: Mapping[str, Any]):
    samples = dataset['samples']
    n = len(samples)
    fraction = float(config.get('test_fraction', .2))
    if not 0 < fraction < 1 or n < 8:
        raise ValueError('Need at least 8 samples and test_fraction between zero and one')
    minimum_train = int(config.get('min_train_rows', 2))
    minimum_validation = int(config.get('min_validation_rows', 1))
    test_start = n - max(1, round(n * fraction))
    test = np.arange(test_start, n)
    proposed = np.arange(test_start)
    clock = dataset['price_path']['timestamp'] if 'timestamp' in dataset['price_path'] else None
    kwargs = dict(clock=clock, max_horizon=dataset.get('max_horizon_bars', 1),
                  sequence_length=dataset.get('sequence_length', 1))
    development, final_audit = purge_training_spans(samples, proposed, test, config, **kwargs)
    if len(development) < minimum_train + minimum_validation:
        raise ValueError('Insufficient development coverage after exact final-test purging')
    method = config.get('method', 'exact_span_walk_forward')
    count = int(config.get('folds', 3))
    if count < 1:
        raise ValueError('folds must be positive')
    folds = []
    if method == 'exact_span_walk_forward':
        groups = np.array_split(development, count + 1)
        proposed_folds = [(np.concatenate(groups[:i]), groups[i]) for i in range(1, len(groups))]
    elif method in {'purged_kfold', 'grouped_scenario'}:
        groups = np.array_split(development, count)
        held_groups = 1 if method == 'purged_kfold' else int(config.get('scenario_test_groups', 2))
        if not 1 <= held_groups < count:
            raise ValueError('Held-out group count must be positive and smaller than folds')
        if comb(count, held_groups) > int(config.get('max_scenarios', 100)):
            raise ValueError('Scenario count exceeds max_scenarios; reduce folds/test groups')
        scenarios = list(combinations(range(count), held_groups))
        proposed_folds = [(np.concatenate([g for i, g in enumerate(groups) if i not in chosen]),
                           np.concatenate([groups[i] for i in chosen])) for chosen in scenarios]
    else:
        raise ValueError(f'Unknown exact validation method: {method}')
    for i, (train, validation) in enumerate(proposed_folds):
        if len(validation) < minimum_validation:
            raise ValueError(f'Insufficient validation coverage in fold {i}')
        train, audit = purge_training_spans(samples, train, validation, config, **kwargs)
        if len(train) < minimum_train:
            raise ValueError(f'Insufficient training coverage after purging fold {i}')
        folds.append({'fold': i, 'train': train, 'validation': validation, 'audit': audit})
    return {'method': method, 'folds': folds, 'development': development,
            'gap': np.setdiff1d(proposed, development), 'test': test,
            'purge_bars': 0, 'embargo_bars': int(config.get('embargo_bars', 0)),
            'test_start': test_start, 'development_end': int(development[-1]) + 1,
            'final_boundary_audit': final_audit,
            'path_type': 'grouped_scenarios' if method == 'grouped_scenario' else 'walk_forward' if method == 'exact_span_walk_forward' else 'purged_kfold'}
