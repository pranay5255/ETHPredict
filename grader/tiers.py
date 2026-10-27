"""Predeclared tier windows and exact dependency-purged growing-window blocks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import RESOLUTIONS, DataContractError, utc_ms
from .splits import purge_training_spans


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise DataContractError(f'Duplicate tier field: {key}')
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


@dataclass(frozen=True)
class TierWindow:
    name: str
    start_ms: int
    end_ms: int

    def contains(self, timestamps) -> np.ndarray:
        values = np.asarray(timestamps, dtype=np.int64)
        return (values >= self.start_ms) & (values < self.end_ms)


@dataclass(frozen=True)
class WalkForwardBlock:
    block: int
    train: np.ndarray
    evaluation: np.ndarray
    history: np.ndarray
    audit: dict


def load_tiers(path: str | Path | None = None) -> dict[str, TierWindow]:
    path = Path(path) if path is not None else Path(__file__).with_name('tiers.yaml')
    try:
        config = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
        if set(config) != {'version', 'tiers', 'block_layout'} or type(config['version']) is not int or config['version'] != 1:
            raise DataContractError('Unknown tier configuration')
        layout = config['block_layout']
        expected = {'evaluation_blocks': 5, 'initial_training_groups': 1,
                    'purge': 'lookback_plus_max_horizon', 'embargo': 'sequence_plus_horizon'}
        if layout != expected or type(layout['evaluation_blocks']) is not int or type(layout['initial_training_groups']) is not int:
            raise DataContractError('Unknown block layout')
        if set(config['tiers']) != {'dev', 'shadow', 'holdout'}:
            raise DataContractError('Unknown tier names')
        result = {}
        for name in ['dev', 'shadow', 'holdout']:
            window = config['tiers'][name]
            if set(window) != {'start', 'end'}:
                raise DataContractError('Unknown tier window field')
            start, end = utc_ms(window['start']), utc_ms(window['end'])
            if end <= start:
                raise DataContractError('Empty tier window')
            result[name] = TierWindow(name, start, end)
        if result['dev'].end_ms != result['shadow'].start_ms or result['shadow'].end_ms != result['holdout'].start_ms:
            raise DataContractError('Tier windows must be adjacent and nonoverlapping')
        for value in [result['shadow'].start_ms, result['holdout'].start_ms]:
            stamp = pd.Timestamp(value, unit='ms', tz='UTC')
            if stamp.day != 1 or stamp != stamp.normalize():
                raise DataContractError('Interior tier boundaries must fall on monthly partitions')
        return result
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise DataContractError('Malformed tier configuration') from exc


def build_walk_forward_blocks(clock: pd.DatetimeIndex, *, resolution: str,
                              max_horizon_bars: int, lookback_bars: int,
                              blocks: int = 5) -> list[WalkForwardBlock]:
    """Source-bar clock, six groups: initial fit then five contiguous evaluations.

    Dependency intervals are inclusive [i-lookback, i+horizon]. This conservative
    convention removes exactly lookback+horizon preceding anchors at a boundary.
    Future-side embargo is the vendored sequence-plus-horizon rule; a growing
    training window has no future-side population to remove.
    """
    if resolution not in RESOLUTIONS or type(blocks) is not int or blocks != 5:
        raise DataContractError('v1 requires a known resolution and five blocks')
    if any(type(value) is not int or value < 1 for value in [max_horizon_bars, lookback_bars]):
        raise DataContractError('Declare positive integer lookback and horizon')
    times = pd.DatetimeIndex(pd.to_datetime(clock, utc=True))
    ms = times.as_unit('ns').asi8 // 1_000_000
    interval = RESOLUTIONS[resolution]
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing or (
            times.as_unit('ns').asi8 % (interval * 1_000_000)).any() or (
            len(ms) > 1 and (np.diff(ms) != interval).any()):
        raise DataContractError('Blocks need a complete ordered source-bar clock')
    width = max_horizon_bars + lookback_bars
    if len(times) < (blocks + 1) * (width + 1):
        raise DataContractError('Insufficient initial training after purging')
    anchors = np.arange(len(times))
    samples = pd.DataFrame({'label_span_start_idx': np.maximum(0, anchors - lookback_bars),
                            'label_span_end_idx': anchors + max_horizon_bars})
    groups = np.array_split(anchors, blocks + 1)
    output = []
    for i in range(1, blocks + 1):
        evaluation = groups[i]
        proposed = np.concatenate(groups[:i])
        train, audit = purge_training_spans(samples, proposed, evaluation,
            {'embargo_mode': 'sequence_plus_horizon'}, max_horizon=max_horizon_bars,
            sequence_length=lookback_bars)
        if not len(train) or (train >= evaluation[0]).any():
            raise DataContractError('Invalid growing training population')
        audit.update(purge_width_bars=width, dependency_convention='inclusive [anchor-lookback, anchor+horizon]',
                     training_start_ms=int(ms[train[0]]), training_end_exclusive_ms=int(ms[train[-1]] + interval),
                     evaluation_start_ms=int(ms[evaluation[0]]), evaluation_end_exclusive_ms=int(ms[evaluation[-1]] + interval))
        history = np.arange(max(0, int(evaluation[0]) - lookback_bars), int(evaluation[-1]) + 1)
        output.append(WalkForwardBlock(i - 1, train, evaluation, history, audit))
    return output


def dev_blocks(resolution: str, *, max_horizon_bars: int, lookback_bars: int,
               tiers_path: str | Path | None = None) -> list[WalkForwardBlock]:
    tier = load_tiers(tiers_path)['dev']
    if resolution not in RESOLUTIONS:
        raise DataContractError('Unknown resolution')
    clock = pd.date_range(pd.Timestamp(tier.start_ms, unit='ms', tz='UTC'),
                          pd.Timestamp(tier.end_ms, unit='ms', tz='UTC'),
                          freq=pd.Timedelta(milliseconds=RESOLUTIONS[resolution]), inclusive='left')
    return build_walk_forward_blocks(clock, resolution=resolution,
                                     max_horizon_bars=max_horizon_bars, lookback_bars=lookback_bars)
