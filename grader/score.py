"""Frozen rubric components and fail-closed development economics.

The canonical execution path is env.score_positions. Daily observations are
explicit UTC cashflow sums; no absent or nonfinite input is treated as zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import DataContractError, RESOLUTIONS
from .economic_gate import economic_gate
from .env import ReplayData, score_positions
from .stats import deflated_sharpe_ratio, pnl_per_turnover, sharpe_ratio
from .tiers import _UniqueKeyLoader


COMPONENTS = {'mean_sharpe', 'worst_sharpe', 'deflated_sharpe', 'cost_robustness', 'drawdown', 'pnl_per_turnover'}


def load_rubric(path=None):
    config = yaml.load(Path(path or Path(__file__).with_name('rubric.yaml')).read_text(), Loader=_UniqueKeyLoader)
    expected = {'version', 'frozen_tag', 'calibration', 'seeds', 'integrity', 'economic',
                'components', 'annualization_days', 'bootstrap', 'execution'}
    if not isinstance(config, dict) or set(config) != expected or type(config['version']) is not int or config['version'] != 1:
        raise DataContractError('Unknown rubric fields/version')
    if config['frozen_tag'] != 'grader-v1' or config['calibration'] not in {'pending_baseline_runs', 'complete'}:
        raise DataContractError('Invalid rubric freeze/calibration state')
    seeds = config['seeds']
    if not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2 or any(type(s) is not int or s < 0 or s >= 2**32 for s in seeds):
        raise DataContractError('Rubric needs two unique fixed grader seeds')
    sections = {
        'integrity': {'causality_cutoffs_per_block', 'position_tolerance', 'fast_seconds', 'long_seconds', 'vram_bytes', 'rss_bytes'},
        'economic': {'min_blocks', 'min_block_trades', 'min_trades', 'min_positive_blocks', 'stress_multiple', 'beat_best_baseline'},
        'bootstrap': {'resamples', 'expected_block_days', 'seed'},
        'execution': {'mode', 'fee_bps', 'spread_bps', 'slippage_bps', 'execution'},
    }
    for section, fields in sections.items():
        if not isinstance(config[section], dict) or set(config[section]) != fields:
            raise DataContractError(f'Unknown rubric {section} fields')
    if set(config['execution']['execution']) != {'mode', 'entry_delay_bars'}:
        raise DataContractError('Unknown execution fields')
    components = config['components']
    if not isinstance(components, dict) or set(components) != COMPONENTS:
        raise DataContractError('Unknown scored component')
    for name, settings in components.items():
        if set(settings) != {'weight', 'zero', 'one'} or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in settings.values()):
            raise DataContractError(f'Invalid component map: {name}')
        if settings['weight'] <= 0 or settings['zero'] == settings['one']:
            raise DataContractError('Component weights/maps must be nondegenerate')
    if not np.isclose(sum(c['weight'] for c in components.values()), 1., rtol=0, atol=1e-12):
        raise DataContractError('Component weights must sum to one')
    if config['annualization_days'] != 365.25:
        raise DataContractError('Unknown 24/7 daily annualization')
    # These are acceptance constraints, not calibration knobs.
    fixed = {'causality_cutoffs_per_block': 20, 'position_tolerance': 1e-9,
             'fast_seconds': 600, 'long_seconds': 3600, 'vram_bytes': 22*1000**3, 'rss_bytes': 48*1000**3}
    economic = {'min_blocks': 5, 'min_block_trades': 25, 'min_trades': 150,
                'min_positive_blocks': 3, 'stress_multiple': 2., 'beat_best_baseline': True}
    if config['integrity'] != fixed or config['economic'] != economic or config['bootstrap']['resamples'] != 1000:
        raise DataContractError('Rubric weakens or changes an acceptance constraint')
    return config


@dataclass(frozen=True)
class BlockEvidence:
    block: int
    markets: tuple[str, ...]
    daily: pd.DataFrame
    cashflows: dict
    net_1x_bps: float
    net_2x_bps: float
    turnover: float
    trades: int
    daily_sharpe: float | None
    cashflow_dates: np.ndarray | None = None


def daily_sharpe(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise DataContractError('Daily Sharpe needs a nonempty finite path')
    metric = sharpe_ratio(array, granularity='1d')['per_period']
    return float(metric['value'] * np.sqrt(365.25)) if metric['available'] else None


def block_evidence(data: ReplayData, targets, *, block: int) -> BlockEvidence:
    ordinary, stress = score_positions(data, targets), score_positions(data, targets, stress_multiple=2.)
    # Timestamp each cashflow when its holding interval ends. Final liquidation
    # is at the inclusive last close, before the next tier's funding payment.
    end_clock = np.r_[data.source_times_ms[2:], data.source_times_ms[-1]+RESOLUTIONS[data.resolution]-1]
    events = pd.DataFrame({'timestamp': pd.to_datetime(end_clock, unit='ms', utc=True),
        'net_1x_bps': ordinary['net_bps'].sum(axis=1), 'net_2x_bps': stress['net_bps'].sum(axis=1),
        'turnover': ordinary['turnover'].sum(axis=1), 'trades': ordinary['trades'].sum(axis=1)})
    if events.empty or not np.isfinite(events.drop(columns='timestamp').to_numpy()).all():
        raise DataContractError('Invalid scoring cashflows')
    events['date'] = events.timestamp.dt.floor('D')
    daily = events.groupby('date', sort=True).sum(numeric_only=True)
    expected = pd.date_range(daily.index[0], daily.index[-1], freq='D')
    if not daily.index.equals(expected):
        raise DataContractError('Missing daily accounting observations')
    return BlockEvidence(block, data.markets, daily, ordinary, float(daily.net_1x_bps.sum()),
        float(daily.net_2x_bps.sum()), float(daily.turnover.sum()), int(daily.trades.sum()),
        daily_sharpe(daily.net_1x_bps.to_numpy() / (10000.*len(data.markets))),
        pd.DatetimeIndex(events.date).as_unit('ns').asi8)


def component_score(blocks: list[BlockEvidence], *, trial_sharpes, trial_count: int, rubric=None):
    rubric = rubric or load_rubric()
    if len(blocks) != 5 or {b.block for b in blocks} != set(range(1, 6)) or len({b.markets for b in blocks}) != 1:
        raise DataContractError('Score needs five distinct comparable declared-market blocks')
    blocks = sorted(blocks, key=lambda b: b.block)
    markets = len(blocks[0].markets)
    daily = pd.concat([b.daily for b in blocks]).groupby(level=0).sum()
    if not np.isfinite(daily.to_numpy()).all() or type(trial_count) is not int or trial_count < 1:
        raise DataContractError('Invalid trial accounting/daily evidence')
    returns = daily.net_1x_bps.to_numpy() / (10000.*markets)
    dsr = deflated_sharpe_ratio(returns, trial_sharpes, trial_count)
    net1 = float(sum(b.net_1x_bps for b in blocks));net2 = float(sum(b.net_2x_bps for b in blocks))
    turnover = float(sum(b.turnover for b in blocks))
    efficiency = pnl_per_turnover(net1, turnover)
    cash = np.concatenate([b.cashflows['net_bps'].sum(axis=1) for b in blocks]) / (10000.*markets)
    equity = np.r_[0., np.cumsum(cash)]
    drawdown = float(np.max(np.maximum.accumulate(equity)-equity))
    sharpes = [b.daily_sharpe for b in blocks]
    available_sharpes = all(value is not None and np.isfinite(value) for value in sharpes)
    raw = {'mean_sharpe': float(np.mean(sharpes)) if available_sharpes else None,
           'worst_sharpe': float(min(sharpes)) if available_sharpes else None,
           'deflated_sharpe': float(dsr['value']) if dsr['available'] else None,
           'cost_robustness': float(net2/net1) if net1 != 0 else None,
           'drawdown': drawdown,
           'pnl_per_turnover': float(efficiency['value']) if efficiency['available'] else None}
    mapped, unavailable = {}, []
    for name, settings in rubric['components'].items():
        value = raw[name]
        if value is None:
            mapped[name] = 0.
            unavailable.append(name)
        elif not np.isfinite(value):
            raise DataContractError(f'Nonfinite rubric component: {name}')
        else:
            mapped[name] = float(np.clip((value-settings['zero'])/(settings['one']-settings['zero']),0.,1.))
    score = float(sum(rubric['components'][name]['weight']*value for name,value in mapped.items()))
    return {'pre_gate_score': score, 'raw_components': raw, 'components': mapped,
            'unavailable_components': unavailable, 'deflated_sharpe_evidence': dsr,
            'net_1x_bps': net1, 'net_2x_bps': net2, 'turnover': turnover,
            'trades': int(sum(b.trades for b in blocks)), 'trial_count': trial_count,
            'trial_sharpes': list(trial_sharpes), 'daily_sharpe_per_period':
                sharpe_ratio(returns, granularity='1d')['per_period'],
            'annualization': '365.25 UTC days; fixed gross notional; partial edge days retained'}


def economic_report(blocks, *, best_baseline_net_bps, engineered_features=(), importance=None, rubric=None):
    rubric = rubric or load_rubric()
    settings = {**rubric['economic'], 'enabled': True, 'require_credible_features': bool(engineered_features),
                'feature_prefix': ''}
    net1 = float(sum(b.net_1x_bps for b in blocks));net2 = float(sum(b.net_2x_bps for b in blocks))
    rows = [{'net_pnl': b.net_1x_bps, 'trades': b.trades} for b in blocks]
    report = economic_gate({'net_pnl': net1, 'trades': sum(b.trades for b in blocks)}, rows,
        {'rows': [{'cost_multiple': 2., 'net_pnl': net2}]}, importance or {}, settings)
    comparable = len(blocks) == 5 and {b.block for b in blocks} == set(range(1,6)) and len({b.markets for b in blocks}) == 1
    report['gates']['comparable_blocks'] = comparable
    report['gates']['beats_best_baseline'] = (best_baseline_net_bps is not None and
        np.isfinite(best_baseline_net_bps) and net1 > best_baseline_net_bps)
    report['passed'] = bool(all(report['gates'].values()))
    report['status'] = 'passed' if report['passed'] else 'rejected'
    report['failures'] = [name for name, passed in report['gates'].items() if not passed]
    report['reason'] = ', '.join(report['failures']) if report['failures'] else 'all predeclared gates passed'
    report['best_baseline_net_bps'] = best_baseline_net_bps
    return report
