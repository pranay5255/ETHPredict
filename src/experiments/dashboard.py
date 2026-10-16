"""Self-contained local experiment evidence dashboard; no service or credentials."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dashboard_payload(report):
    trials = []
    for trial in report.get('trials', []):
        item = {key: trial.get(key) for key in ['trial_id', 'status', 'reason', 'overrides', 'metrics', 'meta_labeler',
                'cost_model', 'runtime', 'feature_importance', 'cost_stress', 'test_evaluation', 'failure_modes', 'artifact_paths']}
        item['roles'] = [role for role, name in [('raw_best', 'raw_best_trial_id'), ('trade_qualified_best', 'best_trading_trial_id'),
                          ('classification_best', 'classification_best_trial_id'), ('calibration_best', 'calibration_best_trial_id')]
                         if report.get('selection', {}).get(name) == trial.get('trial_id')]
        item['diagnostics'] = trial.get('diagnostics', {})
        item['trades'] = {}
        for split in ['validation', 'test']:
            key = 'validation_trades' if split == 'validation' else 'test_trades'
            path = trial.get('artifact_paths', {}).get(key)
            if path and Path(path).is_file():
                frame = pd.read_parquet(path)
                item['trades'][split] = json.loads(frame.to_json(orient='records', date_format='iso'))
            else:
                item['trades'][split] = []
        # Keep the detailed fold artifacts on disk; avoid duplicating huge histories in the HTML payload.
        importance = item.get('feature_importance') or {}
        item['feature_importance'] = {key: importance.get(key) for key in ['status', 'scope', 'summary', 'noise_control_failures', 'auxiliary_comparisons']}
        item['feature_importance']['ablations'] = [{key: row.get(key) for key in ['method', 'feature', 'fold', 'scores', 'change_from_full', 'retained_columns']}
                                                   for fold in importance.get('folds', []) for row in fold.get('ablations', [])]
        trials.append(item)
    return _clean({'run_id': report['run_id'], 'device': report.get('device'), 'evidence_grade': report.get('evidence_grade'),
                   'selection': report.get('selection'), 'trial_accounting': report.get('trial_accounting'),
                   'search_plan': report.get('search_plan'), 'budget': report.get('research_comparison_budget'),
                   'identity': report.get('run_identity'), 'pbo': report.get('pbo'), 'trials': trials,
                   'default_trial': (report.get('best_trading_trial') or report.get('best_trial') or {}).get('trial_id')})


def build_dashboard(report, output=None):
    output = Path(output or Path(report['run_dir']) / 'report' / 'dashboard.html')
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dashboard_payload(report), allow_nan=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('dashboard_template.html').read_text()
    output.write_text(template.replace('/*PLOTLY_LIBRARY*/', get_plotlyjs()).replace('/*EXPERIMENT_DATA*/{}', payload))
    return output


def main():
    parser = argparse.ArgumentParser(description='Build an offline experiment dashboard from a saved run manifest')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(build_dashboard(json.loads(args.manifest.read_text()), args.output))


if __name__ == '__main__':
    main()
