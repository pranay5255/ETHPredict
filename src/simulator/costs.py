"""One explicit simulation contract for causal cost quotes and realized cashflows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.side_data import DataCoverageError


class CostModel:
    def __init__(self, config, price_path=None, feature_frame=None, *, data_dir='data', granularity='5m'):
        self.config = dict(config or {})
        self.mode = self.config.get('mode', 'static')
        if self.mode not in {'static', 'stress', 'dynamic'}:
            raise ValueError(f'Unknown cost mode: {self.mode}')
        self.execution = self.config.get('execution', {}) or {}
        self.execution_mode = self.execution.get('mode', 'taker')
        if self.execution_mode not in {'taker', 'maker', 'hybrid'}:
            raise ValueError(f'Unknown execution mode: {self.execution_mode}')
        self.fill_policy = self.execution.get('fill_policy', 'expected_haircut')
        if self.fill_policy not in {'expected_haircut', 'sampled'}:
            raise ValueError('fill_policy must be expected_haircut or sampled')
        self.delay = int(self.execution.get('entry_delay_bars', 0))
        if self.delay < 0:
            raise ValueError('entry_delay_bars cannot be negative')
        self.path = price_path.reset_index(drop=True) if price_path is not None else None
        self.features = feature_frame.reset_index(drop=True) if feature_frame is not None else None
        self.granularity = granularity
        self.funding = None
        self.source_manifest = []
        for key in ['fee_bps', 'spread_bps', 'slippage_bps', 'stress_multiplier']:
            value = float(self.config.get(key, 1. if key == 'stress_multiplier' else 0.))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.mode == 'dynamic' and self.config.get('dynamic_funding', {}).get('enabled'):
            self._load_funding(self.config['dynamic_funding'], Path(data_dir))
        self.config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        identity = json.dumps({'configuration': self.config, 'sources': self.source_manifest}, sort_keys=True)
        self.identity = hashlib.sha256(identity.encode()).hexdigest()

    def timestamp(self, index):
        if self.path is None or not 0 <= int(index) < len(self.path):
            return None
        row = self.path.iloc[int(index)]
        if 'close_time' in row:
            return pd.to_datetime(row['close_time'], unit='ms', utc=True)
        if 'timestamp' in row:
            return pd.to_datetime(row['timestamp'], utc=True)
        return None

    def _load_funding(self, source, data_dir):
        if source.get('historical') is not True or not source.get('paths'):
            raise ValueError('Dynamic funding requires explicit historical source paths')
        if source.get('rate_unit') not in {'fraction', 'percent', 'bps'}:
            raise ValueError('Dynamic funding rate_unit must explicitly be fraction, percent or bps')
        frames = []
        for value in source['paths']:
            path = Path(value)
            path = path if path.is_absolute() else data_dir / path
            if 'latest' in path.stem.lower():
                raise ValueError('Latest funding snapshots are not historical cashflows')
            frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
            timestamp = pd.to_datetime(frame[source.get('timestamp_column', 'timestamp')],
                                       unit=source.get('timestamp_unit'), utc=True, errors='coerce')
            rate = pd.to_numeric(frame[source.get('rate_column', 'funding_rate')], errors='coerce')
            if source.get('direction_column'):
                # Canonical sign: positive means longs pay shorts.
                direction = frame[source['direction_column']].str.lower()
                if not direction.isin(['long', 'short']).all():
                    raise ValueError('Funding direction must be long or short')
                semantics = source.get('direction_semantics', 'payer')
                if semantics not in {'payer', 'recipient'}:
                    raise ValueError('direction_semantics must be payer or recipient')
                positive = direction.eq('long' if semantics == 'payer' else 'short')
                rate = rate.abs() * np.where(positive, 1., -1.)
            rate = rate * {'fraction': 10000., 'percent': 100., 'bps': 1.}[source['rate_unit']]
            frames.append(pd.DataFrame({'timestamp': timestamp, 'rate_bps': rate}))
            self.source_manifest.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
        funding = pd.concat(frames, ignore_index=True)
        if funding.timestamp.isna().any() or not np.isfinite(funding.rate_bps).all():
            raise ValueError('Invalid historical funding cashflows')
        if funding.groupby('timestamp').rate_bps.nunique().gt(1).any():
            raise ValueError('Conflicting funding cashflows at one timestamp')
        funding = funding.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        if len(funding) < 2:
            raise ValueError('Dynamic funding requires at least two cashflows')
        self.funding = funding
        self.funding_times = funding.timestamp.astype('datetime64[ns, UTC]').astype('int64').to_numpy()
        self.funding_cumulative = np.r_[0., funding.rate_bps.to_numpy(dtype=float).cumsum()]
        period = pd.Timedelta(hours=float(source.get('payment_period_hours', 1.)))
        if pd.isna(period) or period <= pd.Timedelta(0):
            raise ValueError('Funding payment_period_hours must be positive and finite')
        # A tolerance cannot authorize skipping a scheduled funding payment.
        gap = min(pd.Timedelta(source.get('max_gap', period)), period)
        if pd.isna(gap) or gap <= pd.Timedelta(0):
            raise ValueError('Funding max_gap must be positive and finite')
        if (funding.timestamp.diff().dropna() > gap).any():
            raise DataCoverageError('Historical funding has an excessive gap', {'max_gap': str(gap)})
        if self.path is not None:
            start, end = self.timestamp(0), self.timestamp(len(self.path)-1)
            if start is None or end is None or funding.timestamp.iloc[0] > start or funding.timestamp.iloc[-1] < end.floor(period):
                raise DataCoverageError('Historical funding does not cover the price path', {'start': str(start), 'end': str(end)})
        self.funding_source = source

    def _hours(self, start, horizon, exit_index=None):
        entry_time = self.timestamp(start)
        end_time = self.timestamp(exit_index) if exit_index is not None else None
        if entry_time is not None and end_time is not None:
            return max(0., (end_time - entry_time).total_seconds() / 3600.)
        # Estimate from completed past bars only, including for activity clocks.
        if self.path is not None and start > 0:
            previous = self.timestamp(max(0, start - min(start, 32)))
            if entry_time is not None and previous is not None:
                return max(0., (entry_time - previous).total_seconds() / 3600. / min(start, 32) * horizon)
        from src.data.features_all import bars_for_duration
        return horizon / bars_for_duration(self.granularity, hours=1)

    def _state(self, index, signal):
        state = dict(signal)
        if self.features is not None and 0 <= int(index) < len(self.features):
            state.update(self.features.iloc[int(index)].to_dict())
        if self.path is not None and 0 <= int(index) < len(self.path):
            state.update(self.path.iloc[int(index)].to_dict())
        return state

    def _spread(self, state):
        settings = self.config.get('dynamic_spread', {}) or {}
        if self.mode != 'dynamic' or not settings.get('enabled'):
            return float(self.config.get('spread_bps', 0.)), 'configured_static'
        source = settings.get('source', 'high_low_proxy')
        if source == 'order_book':
            value = float(state.get('spread_bps', np.nan))
            if state.get('order_book_missing', 0) or not np.isfinite(value) or value < 0:
                raise DataCoverageError('Historical spread unavailable', {'source': source})
            return value, source
        if source != 'high_low_proxy':
            raise ValueError(f'Unknown spread source: {source}')
        high, low = float(state.get('high', np.nan)), float(state.get('low', np.nan))
        if not np.isfinite([high, low]).all() or low <= 0 or high < low:
            raise DataCoverageError('OHLC spread proxy unavailable', {'source': source})
        estimate = np.log(high / low) * 10000. * float(settings.get('range_fraction', .1))
        return max(float(settings.get('floor_bps', 0.)), estimate), 'controlled_high_low_proxy'

    def quote(self, signal, *, exit_index=None):
        start, horizon = int(signal.get('entry_idx', signal.get('sample_index', 0))), int(signal.get('horizon_bars', 1))
        # Expected quotes are anchored at the signal bar, never the delayed future entry.
        anchor = int(signal.get('sample_index', start)) if exit_index is None else start
        entry = self._state(anchor, signal)
        spread, spread_source = self._spread(entry)
        if exit_index is not None and self.mode == 'dynamic':
            exit_spread, _ = self._spread(self._state(exit_index, signal))
            spread = (spread + exit_spread) / 2.
        taker = float(self.config.get('fee_bps', 0.))
        maker = float(self.execution.get('maker_fee_bps', taker))
        fee = 2*taker if self.execution_mode == 'taker' else 2*maker if self.execution_mode == 'maker' else maker+taker
        if self.execution_mode == 'maker':
            spread = 0.  # No spread capture is credited; adverse selection remains unmodeled.
        elif self.execution_mode == 'hybrid':
            spread *= .5
        slippage = float(self.config.get('slippage_bps', 0.))
        settings = self.config.get('dynamic_slippage', {}) or {}
        if self.mode == 'dynamic' and settings.get('enabled'):
            vol_bps = abs(float(entry.get('realized_vol', 0.))) * 10000.
            notional = float(signal.get('position_notional', self.config.get('reference_notional', 1000.)))
            depth = float(entry.get('bid_depth' if int(signal.get('side', 1)) < 0 else 'ask_depth', 0.))
            close = float(entry.get('close', 0.))
            impact = notional / (depth * close) if depth > 0 and close > 0 else 0.
            if float(settings.get('depth_coefficient_bps', 0.)) and (depth <= 0 or entry.get('order_book_missing', 0)):
                raise DataCoverageError('Depth-dependent slippage requires historical depth', {})
            slippage += (float(settings.get('volatility_coefficient', 0.)) * vol_bps
                         + float(settings.get('spread_coefficient', 0.)) * spread
                         + float(settings.get('depth_coefficient_bps', 0.)) * impact
                         + float(settings.get('size_coefficient_bps', 0.)) * notional / max(float(settings.get('reference_notional', 1000.)), 1e-12))
        side = int(signal.get('side', 0))
        hours = self._hours(anchor, horizon, exit_index)
        funding = float(self.config.get('funding_bps_per_hour', 0.)) * hours
        funding_source = 'configured_static'
        if self.funding is not None:
            funding_source = 'historical_settled_cashflows' if exit_index is not None else 'last_known_payment_estimate'
            begin = self.timestamp(start if exit_index is not None else anchor)
            if begin is None:
                raise ValueError('Historical funding requires an explicit timestamp clock')
            if exit_index is not None:
                end = self.timestamp(exit_index)
                left = int(np.searchsorted(self.funding_times, begin.value, side='right'))
                right = int(np.searchsorted(self.funding_times, end.value, side='right'))
                funding = float(self.funding_cumulative[right] - self.funding_cumulative[left]) * side
            else:
                delay = pd.Timedelta(self.config['dynamic_funding'].get('availability_delay', '0s'))
                if delay < pd.Timedelta(0):
                    raise ValueError('Funding publication delay cannot be negative')
                available = int(np.searchsorted(self.funding_times, (begin - delay).value, side='right')) - 1
                if available < 0:
                    raise DataCoverageError('No funding rate known at candidate time', {'timestamp': str(begin)})
                funding = float(self.funding.rate_bps.iloc[available]) * side * hours / float(self.config['dynamic_funding'].get('payment_period_hours', 1.))
        multiplier = float(self.config.get('stress_multiplier', 1.))
        components = {'fee_bps': fee*multiplier, 'spread_bps': spread*multiplier,
                      'slippage_bps': slippage*multiplier, 'funding_bps': funding*multiplier}
        if not np.isfinite(list(components.values())).all() or min(fee, spread, slippage) < 0:
            raise ValueError('Invalid cost components')
        return {**components, 'total_cost_bps': sum(components.values()), 'cost_model_id': self.identity,
                'cost_model_mode': self.mode, 'cost_config_hash': self.config_hash, 'spread_source': spread_source, 'funding_source': funding_source,
                'simulation_type': 'dynamic_cost_simulation' if self.mode == 'dynamic' else 'static_cost_simulation',
                'execution_mode': self.execution_mode, 'entry_delay_bars': self.delay}

    def fill(self, signal):
        for key in ['maker_fill_probability', 'missed_fill_probability']:
            value = float(self.execution.get(key, 1. if key == 'maker_fill_probability' else 0.))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Fill probabilities must lie in [0, 1]')
        probability = float(self.execution.get('maker_fill_probability', 1.)) if self.execution_mode in {'maker', 'hybrid'} else 1.
        probability *= 1. - float(self.execution.get('missed_fill_probability', 0.))
        if not 0 <= probability <= 1:
            raise ValueError('Fill probabilities must lie in [0, 1]')
        key = f"{self.execution.get('seed', 42)}:{signal.get('sample_index')}:{signal.get('horizon')}"
        uniform = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / 2**64
        filled = uniform < probability if self.fill_policy == 'sampled' else probability > 0
        factor = (1. if filled else 0.) if self.fill_policy == 'sampled' else probability
        return {'fill_probability': probability, 'filled': filled, 'fill_factor': factor, 'fill_policy': self.fill_policy}

    def validate_records(self, records):
        if 'cost_model_id' in records and not records.cost_model_id.eq(self.identity).all():
            raise ValueError('Backtest cashflows came from a different cost model')

    def manifest(self):
        return {'mode': self.mode, 'id': self.identity, 'configuration': self.config, 'sources': self.source_manifest,
                'simulation_type': 'dynamic_cost_simulation' if self.mode == 'dynamic' else 'static_cost_simulation',
                'live_execution_ready': False, 'expected_cost_clock': 'causal signal state only',
                'realized_funding_clock': 'entry exclusive, exit inclusive settled payments',
                'maker_assumption': 'fill haircut; no spread capture credit; adverse selection not calibrated'}


def dataset_cost_model(dataset, config):
    if 'cost_model' not in dataset:
        dataset['cost_model'] = CostModel(config.get('costs', {}), dataset.get('price_path'), dataset.get('feature_frame'),
                                          data_dir=config.get('data', {}).get('dir', 'data'), granularity=dataset.get('granularity', '5m'))
    return dataset['cost_model']


def configured_notional(config):
    alpha = config.get('alpha_backtest', {}) or {}
    notional = float(alpha.get('position_notional', alpha.get('initial_capital', 100000.)))
    cap = float(alpha.get('max_position_notional', notional))
    return min(notional, cap) if cap > 0 else notional
