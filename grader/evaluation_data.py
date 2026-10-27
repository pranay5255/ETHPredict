"""Verified tier input, causal lake views and shared execution preparation."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .contracts import Declaration
from .costs import CostModel
from .data import CANDLE_COLUMNS, DATASETS, FUNDING_COLUMNS, RESOLUTIONS, DataContractError, LakeLoader, LoadedDataset, MarketData
from .env import prepare_replay
from .score import load_rubric
from .tiers import WalkForwardBlock, build_walk_forward_blocks, load_tiers
from .splits import purge_training_spans


def _stamp(value):
    return pd.Timestamp(value, unit='ms', tz='UTC')


@dataclass
class EvaluationData:
    datasets: dict[str, dict[str, LoadedDataset]]
    resolution: str
    tier: str
    start_ms: int
    end_ms: int

    @classmethod
    def load(cls, root, declaration, *, tier='dev', private_root=None):
        if tier not in {'dev', 'shadow'}:
            raise DataContractError('Holdout evaluation is not an automated grader operation')
        tiers = load_tiers()
        windows = [(root, tiers['dev'].start_ms, tiers['dev'].end_ms)]
        if tier == 'shadow':
            if private_root is None:
                raise DataContractError('Private service must provision the shadow lake')
            windows.append((private_root, tiers['shadow'].start_ms, tiers['shadow'].end_ms))
        result = {}
        for market in declaration.markets:
            result[market] = {}
            selected = set(DATASETS) if declaration.uses_dev_lake else {f'candles_{declaration.resolution}', 'fundings_1h'}
            for dataset in sorted(selected):
                parts = []
                for path, start, end in windows:
                    loaded = LakeLoader(path).load(market, dataset, DATASETS[dataset], _stamp(start), _stamp(end), require_complete=False)
                    permitted_gap = dataset == 'mark_price_5m' or dataset == 'fundings_1h'
                    if any(gap['kind'] != 'before_source_start' or not permitted_gap for gap in loaded.provenance['gaps']):
                        raise DataContractError(f'Incomplete required source: {market}/{dataset}')
                    parts.append(loaded)
                frame = pd.concat([part.frame for part in parts], ignore_index=True)
                if not frame.timestamp_ms.is_unique:
                    raise DataContractError('Tier source windows overlap')
                provenance = {'market': market, 'market_id': parts[0].provenance['market_id'],
                    'dataset': dataset, 'resolution': DATASETS[dataset],
                    'sources': [part.provenance for part in parts]}
                result[market][dataset] = LoadedDataset(frame, provenance)
        return cls(result, declaration.resolution, tier, tiers[tier].start_ms, tiers[tier].end_ms)

    @property
    def candles(self):
        return MarketData({market: datasets[f'candles_{self.resolution}'] for market, datasets in self.datasets.items()})

    def blocks(self, declaration):
        frame = next(iter(self.candles.datasets.values())).frame
        times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp_ms, unit='ms', utc=True))
        for item in self.candles.datasets.values():
            if not np.array_equal(item.frame.timestamp_ms.to_numpy(), frame.timestamp_ms.to_numpy()):
                raise DataContractError('Declared markets require identical complete source clocks')
        # Reserve the final source bar for execution/liquidation, before tier end.
        anchors = times[:-1]
        if self.tier == 'dev':
            return build_walk_forward_blocks(anchors, resolution=self.resolution,
                max_horizon_bars=declaration.max_horizon_bars, lookback_bars=declaration.lookback_bars)
        # Shadow already has the complete dev training prelude. Split all private
        # decision anchors into five contiguous evaluation groups, growing fit.
        ms = anchors.as_unit('ms').asi8
        width = declaration.lookback_bars + declaration.max_horizon_bars
        samples = pd.DataFrame({'label_span_start_idx': np.arange(len(ms))-declaration.lookback_bars,
                                'label_span_end_idx': np.arange(len(ms))+declaration.max_horizon_bars})
        groups = np.array_split(np.flatnonzero(ms >= self.start_ms), 5)
        output = []
        for number, evaluation in enumerate(groups):
            if len(evaluation) < 21:
                raise DataContractError('Insufficient shadow decisions for integrity checks')
            train, audit = purge_training_spans(samples, np.arange(evaluation[0]), evaluation,
                {'embargo_mode': 'sequence_plus_horizon'}, max_horizon=declaration.max_horizon_bars,
                sequence_length=declaration.lookback_bars)
            if not len(train):
                raise DataContractError('Shadow purge removed every training anchor')
            audit.update(purge_width_bars=width, training_start_ms=int(ms[train[0]]),
                         training_end_exclusive_ms=int(ms[train[-1]]+RESOLUTIONS[self.resolution]),
                         evaluation_start_ms=int(ms[evaluation[0]]),
                         evaluation_end_exclusive_ms=int(ms[evaluation[-1]]+RESOLUTIONS[self.resolution]))
            history = np.arange(max(0, evaluation[0]-declaration.lookback_bars), evaluation[-1]+1)
            output.append(WalkForwardBlock(number, train, evaluation, history, audit))
        return output

    def market_slice(self, indices):
        return MarketData({market: LoadedDataset(item.frame.iloc[indices].copy().reset_index(drop=True), item.provenance)
                           for market, item in self.candles.datasets.items()})

    def replay(self, source: MarketData, directory, *, training=False):
        directory = Path(directory)
        directory.mkdir(parents=True)
        frames, models = {}, {}
        rubric = load_rubric()
        interval = RESOLUTIONS[self.resolution]
        for market, item in source.datasets.items():
            funding = self.datasets[market]['fundings_1h'].frame
            frame = item.frame.copy()
            if training:
                # The first collected funding payment is after the first candle.
                # Preserve raw training bars, but offer replay only once funding
                # is observed; do not invent the missing opening payment.
                first = int(funding.timestamp_ms.iloc[0])
                frame = frame[frame.timestamp_ms+interval-1 >= first].reset_index(drop=True)
            begin, finish = int(frame.timestamp_ms.iloc[0]), int(frame.timestamp_ms.iloc[-1]+interval-1)
            cashflows = funding[(funding.timestamp_ms >= begin//3_600_000*3_600_000) & (funding.timestamp_ms <= finish)]
            path = directory / f'{market}-funding.parquet'
            cashflows[FUNDING_COLUMNS].to_parquet(path, index=False)
            config = deepcopy(rubric['execution'])
            config['dynamic_funding'] = {'enabled': True, 'historical': True, 'paths': [str(path)],
                'timestamp_column': 'timestamp_ms', 'timestamp_unit': 'ms', 'rate_column': 'rate',
                'rate_unit': 'percent', 'direction_column': 'direction', 'direction_semantics': 'payer',
                'payment_period_hours': 1, 'max_gap': '1h'}
            frames[market] = frame
            models[market] = CostModel(config, frame.assign(close_time=frame.timestamp_ms+interval-1), granularity=self.resolution)
        return prepare_replay(frames, models, resolution=self.resolution)

    def view(self, directory, cutoff):
        """Rebuild a fixed-path read-only lake view for this call's availability."""
        root = Path(directory)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        for market, datasets in self.datasets.items():
            manifest = {'symbol': market, 'market_id': next(iter(datasets.values())).provenance['market_id'], 'datasets': {}}
            for dataset, item in datasets.items():
                selected = item.frame[item.frame.available_at_ms <= cutoff]
                columns = FUNDING_COLUMNS if dataset == 'fundings_1h' else CANDLE_COLUMNS
                selected = selected[columns]
                target = root / market / dataset
                target.mkdir(parents=True)
                hashes = {}
                # Empty early partitions preserve explicit pre-source unavailable
                # windows; no synthetic observation or feature value is added.
                first = pd.Timestamp(next(iter(self.candles.datasets.values())).frame.timestamp_ms.iloc[0], unit='ms')
                last = pd.Timestamp(cutoff, unit='ms')
                for month in pd.period_range(first.to_period('M'), last.to_period('M'), freq='M'):
                    part = selected[pd.to_datetime(selected.timestamp_ms, unit='ms').dt.to_period('M').eq(month)]
                    path = target / f'{month}.parquet'
                    part.to_parquet(path, index=False)
                    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                # For an entirely unavailable mark series, source bounds are
                # metadata only; the files contain no future rows.
                known = item.frame.iloc[:1] if selected.empty else selected
                manifest['datasets'][dataset] = {'expected_interval_ms': RESOLUTIONS[DATASETS[dataset]],
                    'first': _stamp(int(known.timestamp_ms.iloc[0])).isoformat(),
                    'last': _stamp(int(known.timestamp_ms.iloc[-1])).isoformat(),
                    'partitions': hashes, 'rows': len(selected)}
            (root / market / '_manifest.json').write_text(json.dumps(manifest, allow_nan=False))
        return root
