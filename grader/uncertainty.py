"""Stationary daily block bootstrap for the strict numeric keep rule."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import DataContractError
from .score import component_score, daily_sharpe, load_rubric


def stationary_indices(length, expected_block_days, rng):
    if length < 1 or not np.isfinite(expected_block_days) or expected_block_days < 1:
        raise DataContractError('Invalid stationary bootstrap span')
    index = np.empty(length, dtype=int)
    index[0] = rng.integers(length)
    for i in range(1, length):
        index[i] = rng.integers(length) if rng.random() < 1./expected_block_days else (index[i-1]+1)%length
    return index


def score_standard_error(seed_blocks, *, trial_sharpes, trial_count, rubric=None):
    """Synchronous resamples across seeds, costs, turnover and intraday paths.

    Resample consecutive UTC days inside each predeclared evaluation block.
    Intraday cashflows travel with their day, preserving their drawdown geometry.
    All 1,000 replicates use the current ledger count and fixed rubric maps.
    """
    rubric = rubric or load_rubric()
    if len(seed_blocks) != 2 or any(len(blocks) != 5 for blocks in seed_blocks):
        raise DataContractError('Bootstrap needs both seeds and all five blocks')
    settings = rubric['bootstrap']
    rng = np.random.default_rng(settings['seed'])
    for left, right in zip(*seed_blocks):
        if not left.daily.index.equals(right.daily.index):
            raise DataContractError('Seed paths are not on identical UTC days')
    chunks = []
    for blocks in seed_blocks:
        per_seed = []
        for block in blocks:
            if block.cashflow_dates is None:
                raise DataContractError('Missing intraday/day accounting association')
            clock = block.daily.index.as_unit('ns').asi8
            per_seed.append([np.flatnonzero(block.cashflow_dates == day) for day in clock])
        chunks.append(per_seed)
    scores = np.empty(settings['resamples'])
    for replicate in range(settings['resamples']):
        selections = [stationary_indices(len(block.daily), settings['expected_block_days'], rng) for block in seed_blocks[0]]
        seed_scores = []
        for seed, blocks in enumerate(seed_blocks):
            resampled = []
            for i, block in enumerate(blocks):
                selection = selections[i]
                daily = block.daily.iloc[selection].copy()
                daily.index = block.daily.index
                cash_indices = np.concatenate([chunks[seed][i][day] for day in selection])
                cash = {name: values[cash_indices] for name, values in block.cashflows.items()}
                resampled.append(replace(block, daily=daily, cashflows=cash,
                    net_1x_bps=float(daily.net_1x_bps.sum()), net_2x_bps=float(daily.net_2x_bps.sum()),
                    turnover=float(daily.turnover.sum()), trades=int(daily.trades.sum()),
                    daily_sharpe=daily_sharpe(daily.net_1x_bps.to_numpy()/(10000.*len(block.markets)))))
            seed_scores.append(component_score(resampled, trial_sharpes=trial_sharpes,
                                                trial_count=trial_count, rubric=rubric)['pre_gate_score'])
        scores[replicate] = np.mean(seed_scores)
    return {'standard_error': float(scores.std(ddof=1)), 'resamples': int(settings['resamples']),
            'expected_block_days': float(settings['expected_block_days']), 'seed': int(settings['seed']),
            'method': 'stationary UTC daily resampling within each block; seeds synchronized; intraday paths retained',
            'mean': float(scores.mean()), 'minimum': float(scores.min()), 'maximum': float(scores.max())}
