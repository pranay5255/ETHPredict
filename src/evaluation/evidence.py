"""Interpretation gates for reports; availability of a statistic is not evidence."""
import math


def assess_evidence(report, config):
    best = report.get('best_trading_trial') or report.get('best_trial') or {}
    validation = best.get('metrics', {}).get('validation', {})
    test = best.get('metrics', {}).get('test', {})
    stats = test.get('statistics', {})
    def supported(stat, bound, lower=True):
        value = stat.get('value') if isinstance(stat, dict) else None
        return bool(stat.get('available') and isinstance(value, (int, float)) and math.isfinite(value)
                    and (value >= bound if lower else value <= bound)) if isinstance(stat, dict) else False
    gates = {'profitable_validation': validation.get('net_pnl', 0.) > 0,
             'profitable_final_test': test.get('net_pnl', 0.) > 0,
             'minimum_test_trades': test.get('trades', 0) >= int(config.get('pipeline', {}).get('min_validation_trades', 1)),
             'psr_at_least_095': supported(stats.get('psr', {}), .95),
             'dsr_at_least_095': supported(stats.get('dsr', {}), .95),
             'pbo_at_most_010': supported(report.get('pbo', {}), .1, lower=False),
             'single_final_test_use': report.get('final_test_use_count', max([0, *[int((trial.get('final_test_evaluation') or {}).get('count', 0)) for trial in report.get('trials', [])]])) == 1}
    scope = config.get('research', {}).get('evidence_scope', 'research_backtest')
    if report.get('smoke'):
        grade, reason = 'debugging_only', 'Synthetic/smoke software verification'
    elif scope == 'first_bounded_backtest':
        grade, reason = 'exploratory_backtest', 'First bounded study; diagnostic evidence, no alpha or deployment claim'
    elif scope == 'alpha_claim_evaluation' and all(gates.values()):
        grade, reason = 'alpha_claim_grade', 'Predeclared claim evaluation passed all numerical report gates'
    elif gates['profitable_validation'] and gates['minimum_test_trades'] and gates['profitable_final_test']:
        grade, reason = 'validation_promising', 'Positive held-out evidence; full claim gates not established'
    else:
        grade, reason = 'debugging_only', 'Economic or statistical support is missing'
    return {'grade': grade, 'scope': scope, 'gates': gates, 'reason': reason,
            'note': 'No automatic model promotion. Alpha assessment requires a separately predeclared claim study.'}
