"""Frozen validation-outcome haircuts; deliberately distinct from label refits."""
from copy import deepcopy

import numpy as np


def cost_stress_report(candidates, config, *, run_alpha_backtest):
    if 'prediction_role' in candidates and candidates.prediction_role.eq('final_test').any():
        raise ValueError('Cost stress discovery cannot read final-test outcomes')
    multipliers = config.get('costs', {}).get('stress_report_multipliers', [1., 2., 5.])
    rows = []
    for multiplier in multipliers:
        multiplier = float(multiplier)
        if multiplier < 0 or not np.isfinite(multiplier):
            raise ValueError('Cost stress multipliers must be finite and nonnegative')
        modified = candidates.copy()
        actual = modified.get('realized_total_cost_bps', modified['total_cost_bps'])
        funding = modified.get('realized_funding_bps', 0.)
        total = stress_cashflows(actual - funding, funding, multiplier)
        modified['realized_total_cost_bps'] = total
        for component in ['fee', 'spread', 'slippage']:
            key = f'realized_{component}_bps'
            if key in modified:
                modified[key] *= multiplier
        if 'realized_funding_bps' in modified:
            modified['realized_funding_bps'] = np.maximum(funding, 0.) * multiplier + np.minimum(funding, 0.)
        modified['label_net_return'] = modified.label_gross_return - total / 10000. * modified.get('fill_factor', 1.)
        metrics, _ = run_alpha_backtest(modified, config)
        rows.append({'cost_multiple': multiplier, 'net_pnl': metrics['net_pnl'], 'gross_pnl': metrics['gross_pnl'],
                     'trades': metrics['trades'], 'fees': metrics['fees'], 'survives': metrics['net_pnl'] > 0,
                     'max_drawdown': metrics['max_drawdown'], 'pnl_per_turnover': metrics.get('pnl_per_turnover'),
                     'break_even_additional_slippage_bps': metrics.get('break_even_additional_slippage_bps')})
    return {'scope': 'held_out_validation_only', 'protocol': 'frozen_candidates_probabilities_policy_and_exit_outcomes_pnl_only',
            'labels_refit': False, 'rows': rows, 'only_optimistic_costs': bool(rows and rows[0]['survives'] and any(not row['survives'] for row in rows[1:])),
            'note': 'Separate supervised comparisons must regenerate labels and refit meta models when cost/execution assumptions change.'}


def stress_cashflows(transaction_cost_bps, funding_bps, multiplier):
    """The source report haircut as an array adapter: never amplify rebates."""
    return (transaction_cost_bps * multiplier + np.maximum(funding_bps, 0.) * multiplier
            + np.minimum(funding_bps, 0.))
