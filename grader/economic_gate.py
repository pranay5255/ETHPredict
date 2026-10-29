"""Fail-closed development economics gate; never consumes final-test evidence."""
import math


def economic_gate(validation, blocks, stress, importance, settings):
    if not settings.get('enabled', False):
        return {'status': 'disabled', 'passed': True, 'scope': 'validation_only'}
    def positive(value):
        return isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    minimum = int(settings.get('min_blocks', 3))
    positives = sum(positive(row.get('net_pnl')) for row in blocks)
    feature_rows = importance.get('summary', [])
    noise = max([0., *[row.get('negative_log_loss_drop', 0.) for row in feature_rows if row.get('category') == 'noise_control']])
    signal = [row for row in feature_rows if row.get('category') == 'positive' and
              row.get('feature', '').startswith(settings.get('feature_prefix', '')) and
              row.get('negative_log_loss_drop', 0.) > noise and row.get('folds', 0) >= minimum]
    stress_multiple = float(settings.get('stress_multiple', 2.))
    stress_rows = [row for row in stress.get('rows', []) if row.get('cost_multiple') == stress_multiple]
    gates = {'enough_blocks': len(blocks) >= minimum,
             'positive_net_expectancy': positive(validation.get('net_pnl')),
             'enough_trades': validation.get('trades', 0) >= int(settings.get('min_trades', 60)),
             'positive_across_blocks': positives >= int(settings.get('min_positive_blocks', 2)),
             'block_trade_coverage': len(blocks) >= minimum and all(row.get('trades', 0) >= int(settings.get('min_block_trades', 10)) for row in blocks),
             'survives_cost_stress': len(stress_rows) == 1 and positive(stress_rows[0].get('net_pnl')),
             'credible_features': importance.get('status') == 'completed' and bool(signal) and
                                  any(row.get('category') == 'noise_control' for row in feature_rows) and
                                  not importance.get('noise_control_failures')}
    if not settings.get('require_credible_features', True):
        gates.pop('credible_features')
    passed = all(gates.values())
    return {'status': 'passed' if passed else 'rejected', 'passed': passed, 'gates': gates,
            'positive_blocks': positives, 'scope': 'purged_development_only',
            'reason': 'all predeclared gates passed' if passed else ', '.join(k for k,v in gates.items() if not v),
            'note': 'Passing authorizes review, not deployment or automatic final-test evaluation'}
