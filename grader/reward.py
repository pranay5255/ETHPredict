"""Net-PnL primitives, in basis points of fixed per-market notional.

Funding inputs are signed, position-weighted settled cashflows (positive paid,
negative received); transaction costs are already turnover-weighted. Callers
preserve those dependencies in the tensor graph. No implicit broadcasting.
"""
from __future__ import annotations

import numpy as np
import torch


def numpy_step_pnl(positions, returns, costs, funding):
    arrays = [np.asarray(value, dtype=np.float64) for value in [positions, returns, costs, funding]]
    if len({value.shape for value in arrays}) != 1 or any(not np.isfinite(value).all() for value in arrays):
        raise ValueError('Reward inputs need matching shapes and finite values')
    if (arrays[2] < 0).any():
        raise ValueError('Transaction costs cannot be negative')
    return arrays[0] * arrays[1] - arrays[2] - arrays[3]


def torch_step_pnl(positions, returns, costs, funding):
    tensors = [positions, returns, costs, funding]
    if any(not isinstance(value, torch.Tensor) for value in tensors) or len({value.shape for value in tensors}) != 1:
        raise ValueError('Reward inputs need matching tensor shapes')
    if len({(value.device, value.dtype) for value in tensors}) != 1 or not positions.is_floating_point():
        raise ValueError('Reward tensors need the same floating dtype and device')
    if any(not bool(torch.isfinite(value).all()) for value in tensors) or bool((costs < 0).any()):
        raise ValueError('Invalid reward evidence')
    return positions * returns - costs - funding


def torch_position_path(targets, fill_factors, delay=0, initial=None):
    """Differentiable execution scan; all N windows/markets run in parallel.

    Shapes are [N,T,M]. Python iterates over the decision clock only; there is
    never a Python loop over N environments. Full fills use a vectorized path.
    """
    if targets.ndim != 3 or fill_factors.shape != targets.shape or type(delay) is not int or delay < 0:
        raise ValueError('Execution expects [N,T,M] arrays and a nonnegative delay')
    if not bool(torch.isfinite(targets).all()) or not bool(torch.isfinite(fill_factors).all()) or (
            bool((targets.abs() > 1).any()) or bool(((fill_factors < 0) | (fill_factors > 1)).any())):
        raise ValueError('Invalid target positions or fill probabilities')
    n, steps, markets = targets.shape
    previous = torch.zeros((n, markets), dtype=targets.dtype, device=targets.device) if initial is None else initial
    if previous.shape != (n, markets) or previous.dtype != targets.dtype or previous.device != targets.device:
        raise ValueError('Invalid initial positions')
    if delay >= steps:
        return previous[:, None, :].expand(-1, steps, -1) + targets * 0
    if bool((fill_factors == 1).all()):
        return torch.cat([previous[:, None, :].expand(-1, delay, -1), targets[:, :steps-delay, :]], dim=1)
    result = []
    for step in range(steps):
        if step >= delay:
            factor = fill_factors[:, step-delay, :]
            previous = previous + factor * (targets[:, step-delay, :] - previous)
        result.append(previous)
    return torch.stack(result, dim=1)


def torch_path_pnl(targets, returns, unit_costs, funding_rates, fill_factors, *, delay=0,
                   exit_costs=None, stress_multiple=1.):
    """Batched objective with delay, fractional fills, turnover and liquidation.

    All inputs [N,T,M] except terminal one-way costs [N,M]. Unit costs are
    per unit absolute position change; rates/returns are in bps. Positions begin
    flat and liquidate after the last interval and its funding payment.
    """
    if any(value.shape != targets.shape for value in [returns, funding_rates, fill_factors]) or (
            unit_costs.shape not in {targets.shape, (*targets.shape, 2)}):
        raise ValueError('Path arrays must have matching [N,T,M] shapes')
    if stress_multiple < 1 or not np.isfinite(stress_multiple):
        raise ValueError('Stress cannot improve transaction costs')
    positions = torch_position_path(targets, fill_factors, delay)
    previous = torch.cat([torch.zeros_like(positions[:, :1, :]), positions[:, :-1, :]], dim=1)
    delta = positions-previous
    unit = unit_costs if unit_costs.ndim == 3 else torch.gather(unit_costs, -1, (delta<0).long().unsqueeze(-1)).squeeze(-1)
    costs = delta.abs() * unit * stress_multiple
    funding = positions * funding_rates
    funding = torch.clamp_min(funding, 0) * stress_multiple + torch.clamp_max(funding, 0)
    if exit_costs is not None:
        if exit_costs.shape not in {targets[:, -1, :].shape, (*targets[:, -1, :].shape, 2)}:
            raise ValueError('Invalid liquidation costs')
        exit_unit = exit_costs if exit_costs.ndim == 2 else torch.gather(exit_costs, -1, (positions[:, -1, :]>0).long().unsqueeze(-1)).squeeze(-1)
        costs = costs.clone()
        costs[:, -1, :] = costs[:, -1, :] + positions[:, -1, :].abs() * exit_unit * stress_multiple
    return torch_step_pnl(positions, returns, costs, funding), positions
