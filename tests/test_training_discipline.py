from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.meta_labeling_mvp import (
    _prediction_metrics, build_multi_horizon_lighter_dataset, predict_base_model, train_base_model,
)
from src.training.discipline import EarlyStopper, bootstrap_rows, training_holdout
from src.validation.splits import interval_coordinates, overlaps
from tests.test_staged_trial import _v2_config, _write_5m_ohlcv


def test_stopping_restores_selected_state():
    model = torch.nn.Linear(1, 1, bias=False)
    stopper = EarlyStopper(patience=2, min_delta=.01)
    stopped = []
    for epoch, loss in enumerate([1., .6, .61, .7, .2], 1):
        with torch.no_grad():
            model.weight.fill_(epoch)
        stopped.append(stopper.update(loss, epoch, model))
        if stopped[-1]:
            break
    stopper.restore(model)
    assert stopper.best_epoch == 2
    assert stopped == [False, False, False, True]
    assert model.weight.item() == 2


@pytest.fixture
def training_data(tmp_path):
    data = tmp_path / 'data'
    _write_5m_ohlcv(data / 'raw' / 'ETHUSDT-5m-lighter-20260328-20260628.csv', rows=180)
    config = _v2_config(data, tmp_path / 'runs')
    config['training'].update(epochs=4, ensemble_seeds=[11], early_stopping={
        'enabled': True, 'patience': 1, 'min_delta': 1e3, 'validation_fraction': .25})
    return build_multi_horizon_lighter_dataset(config), config


def test_base_stopping_uses_purged_training_holdout_and_scaler(training_data):
    dataset, config = training_data
    indices = np.arange(100)
    model, history = train_base_model(dataset, indices, config, torch.device('cpu'))
    audit = history['early_stopping']
    assert audit['status'] == 'completed'
    assert audit['selected_epoch'] == 1 and audit['epochs_run'] == 2
    fit, val = np.array(audit['fit_indices']), np.array(audit['evaluation_indices'])
    starts, ends = interval_coordinates(dataset['samples'])
    assert not overlaps(starts[fit], ends[fit], starts[val], ends[val]).any()
    assert set(fit) | set(val) <= set(indices)
    assert torch.allclose(model.feature_mean.cpu(), dataset['X'][fit].mean((0, 1), keepdim=True))
    changed = deepcopy(dataset)
    changed['y_ret'][100:] = 1000.
    changed['X'][100:] = 1000.
    second, second_history = train_base_model(changed, indices, config, torch.device('cpu'))
    assert second_history == history
    for key, state in model.state_dict().items():
        assert torch.equal(state, second.state_dict()[key])


def test_seed_ensemble_matches_individual_members_deterministically(training_data):
    dataset, config = training_data
    config['training']['ensemble_seeds'] = [11, 29]
    indices, validation = np.arange(100), np.arange(120, 128)
    ensemble, history = train_base_model(dataset, indices, config, torch.device('cpu'))
    aggregate = predict_base_model(ensemble, dataset, validation, torch.device('cpu'))
    repeated, _ = train_base_model(dataset, indices, config, torch.device('cpu'))
    pd.testing.assert_frame_equal(aggregate, predict_base_model(repeated, dataset, validation, torch.device('cpu')))
    predictions = []
    for seed in [11, 29]:
        single = deepcopy(config)
        single['training']['ensemble_seeds'] = [seed]
        member, _ = train_base_model(dataset, indices, single, torch.device('cpu'))
        predictions.append(predict_base_model(member, dataset, validation, torch.device('cpu'))['next_5m_pred_return'])
    assert np.allclose(aggregate.next_5m_pred_return, np.mean(predictions, axis=0))
    assert np.allclose(aggregate.next_5m_pred_return_seed_std, np.std(predictions, axis=0))
    metrics = _prediction_metrics(aggregate, dataset['horizon_names'])
    assert set(metrics['instability']['per_seed']) == {'11', '29'}
    assert metrics['instability']['metric_variance']['next_5m']['return_mse'] >= 0
    assert history['ensemble_seeds'] == [11, 29]
    assert 'members.0.feature_mean' in ensemble.state_dict()


@pytest.mark.parametrize('kind', ['linear', 'tree'])
def test_tabular_baselines_share_prediction_schema(training_data, kind):
    dataset, config = training_data
    config['model']['base']['type'] = kind
    config['model']['base']['n_estimators'] = 5
    model, history = train_base_model(dataset, np.arange(100), config, torch.device('cpu'))
    frame = predict_base_model(model, dataset, np.arange(120, 128), torch.device('cpu'))
    assert frame.shape[0] == 8
    assert np.isfinite(frame.next_5m_pred_return).all()
    assert frame.next_5m_direction_prob.between(0, 1).all()
    assert history['model_type'] == kind
    assert history['early_stopping']['status'] == 'not_applicable'
    assert '_extra_state' in model.state_dict()


@pytest.mark.parametrize('mode', ['none', 'bootstrap', 'sequential_bootstrap'])
def test_training_only_resampling_is_deterministic(mode):
    samples = pd.DataFrame({'label_span_start_idx': [0, 0, 1, 10], 'label_span_end_idx': [3, 3, 4, 12]})
    first, second = bootstrap_rows(samples, mode, 7), bootstrap_rows(samples, mode, 7)
    assert np.array_equal(first, second)
    assert len(first) == len(samples) and set(first) <= set(range(4))


def _indicator_sequential_bootstrap(starts, ends, seed):
    """Brute-force AFML reference over an explicit events-by-bars indicator."""
    rng = np.random.default_rng(seed)
    bars = np.arange(ends.max() + 1)
    indicator = (bars >= starts[:, None]) & (bars <= ends[:, None])
    concurrency, draws = np.zeros(len(bars)), []
    for _ in range(len(starts)):
        uniqueness = np.array([np.mean(1. / (concurrency[row] + 1.)) for row in indicator])
        choice = int(rng.choice(len(starts), p=uniqueness / uniqueness.sum()))
        draws.append(choice)
        concurrency[indicator[choice]] += 1
    return np.asarray(draws)


@pytest.mark.parametrize('seed', [1, 7, 99])
def test_sequential_bootstrap_matches_indicator_reference(seed):
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.integers(0, 120, 40))
    ends = starts + rng.integers(0, 12, 40)
    samples = pd.DataFrame({'label_span_start_idx': starts, 'label_span_end_idx': ends})
    assert np.array_equal(bootstrap_rows(samples, 'sequential_bootstrap', seed),
                          _indicator_sequential_bootstrap(starts, ends, seed))


def test_holdout_keeps_timestamp_groups_together():
    samples = pd.DataFrame({'label_span_start_idx': np.repeat(np.arange(10) * 4, 2),
                            'label_span_end_idx': np.repeat(np.arange(10) * 4 + 1, 2)})
    fit, validation, audit = training_holdout(samples, np.arange(20), {}, {'validation_fraction': .3})
    assert audit['overlap_count_after'] == 0
    assert set(samples.iloc[fit].label_span_start_idx).isdisjoint(samples.iloc[validation].label_span_start_idx)


def test_two_seed_constant_and_logistic_pipeline_smoke(tmp_path):
    import json
    from pathlib import Path
    import yaml
    from src.experiments.staged_trial import run_staged_trial

    data = tmp_path / 'data'
    source = data / 'raw' / 'ETHUSDT-5m-lighter-20260328-20260628.csv'
    _write_5m_ohlcv(source, rows=192)
    raw = pd.read_csv(source, header=None)
    close = 3000. * np.exp(.003*np.sin(np.arange(len(raw))*.65))
    raw[1], raw[2], raw[3], raw[4] = close, close*1.0001, close*.9999, close
    raw.to_csv(source, header=False, index=False)
    config = _v2_config(data, tmp_path / 'runs')
    config['training'].update(ensemble_seeds=[11, 29], early_stopping={'enabled': True, 'patience': 2})
    config['smoke'].update(max_rows=192, max_trials=2)
    config['validation']['method'] = 'exact_span_walk_forward'
    config['research'] = {'final_test_guard': {'evaluate': False}}
    config['benchmark'] = {'enabled': False}
    config['tracking'] = {'trackio': {'enabled': False}}
    config['search'] = {'mode': 'grid', 'max_trials': 2, 'trials': [
        {'id': 'constant', 'overrides': {'model.meta_labeler.type': 'constant'}},
        {'id': 'logistic', 'overrides': {'model.meta_labeler.type': 'logistic'}},
    ]}
    path = tmp_path / 'discipline.yml'
    path.write_text(yaml.safe_dump(config))
    report = run_staged_trial(path, smoke=True, device='cpu', allow_cpu=True)
    assert len(report['trials']) == 2
    for trial in report['trials']:
        assert trial['status'] == 'completed'
        assert trial['test_evaluation']['status'] == 'disabled'
        assert trial['metrics']['test'] == {}
        assert trial['final_training_history']['ensemble_seeds'] == [11, 29]
        frame = pd.read_parquet(trial['artifact_paths']['oof_predictions'])
        assert 'next_5m_pred_return_seed_std' in frame
        assert trial['metrics']['validation']['instability']['per_seed']
        assert set(trial['metrics']['validation']['forecast_baselines']) == {'zero_return', 'momentum'}
        assert Path(trial['artifact_paths']['forecast_diagnostics']).exists()
        diagnostics = json.loads(Path(trial['diagnostics_path']).read_text())
        assert diagnostics['validation']['meta_label_thresholds']
    assert report['trials'][0]['meta_labeler']['fallback_reason'] == 'configured_constant'
    assert report['trials'][1]['meta_labeler']['kind'] == 'logistic'


def test_tiny_stopping_fold_audits_actual_fixed_epoch_fit():
    samples = pd.DataFrame({'label_span_start_idx': [0, 1, 2], 'label_span_end_idx': [2, 3, 4]})
    fit, validation, audit = training_holdout(samples, np.arange(3), {}, {'min_rows': 2})
    assert audit['status'] == 'unavailable'
    assert np.array_equal(fit, np.arange(3)) and not len(validation)
    assert audit['fit_indices'] == [0, 1, 2] and audit['evaluation_indices'] == []
    assert audit['attempted_evaluation_indices']
