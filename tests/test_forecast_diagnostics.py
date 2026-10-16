import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.forecast_diagnostics import calibration_summary, forecast_diagnostics, map_forecast
from src.experiments.forecast_benchmark import run_forecast_benchmark, zero_return_predictions
from src.experiments.meta_labeling_mvp import candidate_signals, research_spec_hash
from tests.test_forecast_benchmark import _synthetic_dataset, _v2_config, _write_5m_ohlcv


@pytest.mark.parametrize('mode,side,eligible', [
    ('raw_sign', 1, True), ('probability_sign', -1, True), ('expected_edge', 1, True),
    ('volatility_scaled_edge', 1, False), ('horizon_agreement', 1, False),
])
def test_mapping_can_change_without_training(mode, side, eligible):
    result = map_forecast(.002, .3, .01, 1., 5., 1., {'mode': mode, 'min_scaled_edge': 1.})
    assert result['side'] == side and result['is_candidate'] == eligible
    assert result['candidate_mapping'] == mode
    assert result['expected_edge_bps'] == pytest.approx(side * 20. - 5.)


def test_mapping_default_cost_filter_zero_signal_and_input_immutability():
    assert not map_forecast(.0001, .7, .001, 0, 5, 0)['is_candidate']
    assert not map_forecast(0, .5, .001, 0, 5, 0, {'mode': 'raw_sign'})['is_candidate']
    dataset = _synthetic_dataset()
    frame = zero_return_predictions(dataset, np.arange(3))
    original = frame.copy(deep=True)
    mapped = candidate_signals(frame, dataset, {'candidate_mapping': {'mode': 'probability_sign'}})
    pd.testing.assert_frame_equal(frame, original)
    assert not mapped.is_candidate.any()
    assert research_spec_hash({'candidate_mapping': {'mode': 'raw_sign'}}) != research_spec_hash({'candidate_mapping': {'mode': 'expected_edge'}})


def test_calibration_and_edge_bucket_values():
    cal = calibration_summary([0, 1], [.2, .8])
    assert cal['brier_score'] == pytest.approx(.04)
    assert cal['ece'] == pytest.approx(.2)
    assert sum(row['count'] for row in cal['reliability']) == 2
    frame = pd.DataFrame({'h_true_return': [-.01, -.005, .005, .01], 'h_pred_return': [-.01, -.005, .005, .01],
                          'h_direction_prob': [.1, .2, .8, .9]})
    metrics = forecast_diagnostics(frame, 'h', cost_bps=5)
    assert metrics['rank_correlation']['spearman'] == pytest.approx(1.)
    assert metrics['rank_correlation']['edge_to_realized_net_spearman'] == pytest.approx(1.)
    assert metrics['return_calibration']['slope'] == pytest.approx(1.)
    rows = metrics['edge_ranking']['predicted_edge_buckets']
    assert sum(row['coverage'] for row in rows) == pytest.approx(1.)
    for row in rows:
        if row['count']:
            assert row['mean_realized_gross_edge_bps'] - row['mean_realized_net_edge_bps'] == pytest.approx(5.)


def test_forecast_only_smoke_expanded_schema_without_final_test(tmp_path):
    data = tmp_path / 'data'
    _write_5m_ohlcv(data / 'raw' / 'ETHUSDT-5m-lighter-20260328-20260628.csv')
    config = _v2_config(data, tmp_path / 'runs')
    config['benchmark'].update(models=['momentum', 'timesfm'], route_to_meta_backtest=False,
                               timesfm={'enabled': False, 'context_length': 32})
    config['research'] = {'final_test_guard': {'evaluate': False}}
    config['tracking'] = {'trackio': {'enabled': False}}
    path = tmp_path / 'forecast.yml'
    path.write_text(yaml.safe_dump(config))
    report = run_forecast_benchmark(path, smoke=True, device='cpu', allow_cpu=True)['benchmark']
    assert report['test_evaluation_count'] == 0
    models = {row['model_id']: row for row in report['models']}
    assert set(models) == {'zero_return', 'momentum', 'timesfm'}
    assert models['timesfm']['configuration']['context_length'] == 32
    assert models['timesfm']['skip_reason']
    for name in ['zero_return', 'momentum']:
        saved = json.loads(Path(models[name]['manifest_path']).read_text())
        assert saved['metrics']['forecast']['test'] == {}
        assert 'backtest' not in saved['metrics']
        metrics = saved['metrics']['forecast']['validation']['next_5m']
        assert {'calibration', 'rank_correlation', 'edge_ranking'} <= metrics.keys()
        if name == 'zero_return':
            assert metrics['edge']['mean_realized_net_edge_bps'] == 0.
            assert metrics['edge_ranking']['mean_realized_net_edge_bps'] == 0.


def test_forecast_only_tabular_and_timesfm_audit(tmp_path, monkeypatch):
    import torch
    import src.experiments.forecast_benchmark as bench
    from src.experiments.meta_labeling_mvp import build_multi_horizon_lighter_dataset
    from tests.test_forecast_benchmark import FakeTimesFM

    data = tmp_path / 'data'
    _write_5m_ohlcv(data / 'raw' / 'ETHUSDT-5m-lighter-20260328-20260628.csv')
    config = _v2_config(data, tmp_path / 'runs')
    config['benchmark'].update(models=['linear', 'tree', 'timesfm'], route_to_meta_backtest=False,
                               timesfm={'context_length': 8})
    config['model']['base']['n_estimators'] = 3
    config['research'] = {'final_test_guard': {'evaluate': False}}
    config['tracking'] = {'trackio': {'enabled': False}}
    adapter = bench.TimesFMZeroShotAdapter
    monkeypatch.setattr(bench, 'TimesFMZeroShotAdapter', lambda **kwargs: adapter(model=FakeTimesFM(), **kwargs))
    dataset = build_multi_horizon_lighter_dataset(config)
    report = bench.run_forecast_benchmark_from_dataset(config, dataset, tmp_path / 'benchmark', device=torch.device('cpu'), smoke=True)
    models = {row['model_id']: row for row in report['models']}
    assert models['timesfm']['status'] == 'success'
    assert models['timesfm']['timesfm']['horizon_alignment'] == {'next_5m': 0, 'next_hour': 2}
    for name in ['linear', 'tree']:
        assert models[name]['final_training_history']['model_type'] == name
        assert models[name]['metrics']['forecast']['validation']['next_5m']['calibration']['count'] > 0
        assert models[name]['metrics']['forecast']['test'] == {}
    assert report['test_evaluation_count'] == 0


def test_trackio_separates_forecast_and_trading_namespaces(monkeypatch, tmp_path):
    import src.experiments.meta_labeling_mvp as mvp
    captured = {}
    monkeypatch.setattr(mvp, 'log_trackio_run', lambda config, **kwargs: captured.update(kwargs))
    trial = {'trial_id': 'one', 'manifest_path': tmp_path / 'manifest.json', 'metrics': {'validation': {'net_pnl': 3., 'forecast': {'h': {'mse': .1}},
             'forecast_baselines': {'zero_return': {}}, 'instability': {'variance': .2}}, 'test': {}}}
    mvp._log_meta_label_trial_trackio({}, run_id='fixture', trial=trial, selection_status='raw_best',
                                    raw_best=trial, best_trading=None, selection={}, trial_count=1, smoke=True)
    assert captured['metrics']['alpha']['validation']['net_pnl'] == 3.
    assert 'forecast' not in captured['metrics']['alpha']['validation']
    assert captured['metrics']['forecast']['validation']['h']['mse'] == .1
    assert captured['metrics']['model_instability']['validation']['variance'] == .2
