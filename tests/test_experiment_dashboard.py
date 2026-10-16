import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.experiments.dashboard import build_dashboard, dashboard_payload
from src.experiments.staged_trial import run_staged_trial
from src.evaluation.backtest_stats import period_pnl_series
from tests.test_staged_trial import _write_5m_ohlcv, _v2_config


def test_realized_exit_cashflows_are_not_credited_at_signal_time():
    clock = pd.date_range('2026-01-01', periods=4, freq='5min')
    candidates = pd.DataFrame({'timestamp': clock, 'bar_close_timestamp': clock + pd.Timedelta('5min') - pd.Timedelta('1ms'),
                               'vertical_barrier_timestamp': clock + pd.Timedelta('10min')})
    exit_time = candidates.bar_close_timestamp.iloc[2]
    trades = pd.DataFrame({'timestamp': [clock[0]], 'pnl_timestamp': [exit_time], 'net_pnl': [10.]})
    pnl = period_pnl_series(candidates, trades)
    assert pnl.iloc[0] == 0 and pnl.loc[exit_time.tz_localize('UTC')] == 10.
    assert pnl.sum() == 10.
    assert len(pnl) == 6  # Common planned horizon tail, including zero-PnL periods.


def test_full_pipeline_search_importance_costs_and_offline_dashboard(tmp_path):
    data = tmp_path / 'data'
    _write_5m_ohlcv(data / 'raw' / 'ETHUSDT-5m-lighter-fixture.csv', rows=160)
    config = _v2_config(data, tmp_path / 'runs')
    config['model']['base']['type'] = 'linear'
    config['model']['meta_labeler']['type'] = 'logistic'
    config['features'] = {'include': ['ohlcv', 'returns']}
    config['feature_importance'] = {'enabled': True, 'noise_features': 1, 'max_base_features': 1,
                                  'max_folds': 1, 'repeats': 1, 'single_feature': True, 'ablation_groups': ['returns']}
    config['search'] = {'mode': 'shuffled_grid', 'seed': 7, 'max_trials': 2,
                        'trials': [{'id': 'normal', 'overrides': {}}, {'id': 'skip', 'overrides': {'trial.skip': True}}]}
    config['smoke']['max_trials'] = 2
    config['research'] = {'final_test_guard': {'evaluate': False}}
    config['benchmark'] = {'enabled': False}
    config['tracking'] = {'trackio': {'enabled': False}}
    config['costs'].update(mode='stress', stress_multiplier=2., execution={'mode': 'hybrid', 'maker_fill_probability': .8})
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump(config))
    report = run_staged_trial(path, smoke=True, device='cpu', allow_cpu=True)
    assert report['trial_accounting']['counts']['skipped'] == 1
    assert report['trial_accounting']['counts']['completed'] == 1
    assert report['search_plan']['seed'] == 7
    trial = next(row for row in report['trials'] if row['status'] == 'completed')
    assert trial['feature_importance']['scope'] == 'purged_outer_validation_only'
    assert trial['cost_model']['mode'] == 'stress'
    assert len(trial['cost_stress']['rows']) == 3
    assert trial['metrics']['test'] == {}
    dashboard = Path(report['dashboard_path'])
    html = dashboard.read_text()
    assert 'Plotly.react' in html and 'const DATA=' in html
    assert '/*EXPERIMENT_DATA*/' not in html and '/*PLOTLY_LIBRARY*/' not in html
    assert 'noise_control' in html and 'Final test is not evaluated' in html
    assert not (Path(report['run_dir']).parent / '_final_test_reuse_ledger.json').exists()
    payload = dashboard_payload(report)
    assert payload['trials'][0]['metrics'] == report['trials'][0]['metrics']
    assert Path(trial['artifact_paths']['feature_importance_table']).is_file()
