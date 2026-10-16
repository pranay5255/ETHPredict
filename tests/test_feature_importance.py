import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.feature_importance import permutation_meta, fold_importance, summarize_importance
from src.experiments.meta_labeling_mvp import fit_meta_labeler, build_multi_horizon_lighter_dataset, build_validation_splits, train_base_model, evaluate_meta_fold, predict_base_model
from tests.test_meta_models import candidate_fixture, meta_config
from tests.test_staged_trial import _write_5m_ohlcv, _v2_config


def test_permutation_separates_signal_from_trained_noise():
    rng = np.random.default_rng(4)
    frame = candidate_fixture(300)
    frame['pred_return'] = rng.normal(size=300)
    frame['meta_label'] = (frame.pred_return > 0).astype(float)
    frame['direction_prob'] = .5
    frame['diagnostic_noise_0'] = rng.normal(size=300)
    config = meta_config()
    config['alpha_backtest']['meta_threshold_selection']['enabled'] = False
    model = fit_meta_labeler(frame.iloc[:200], config, [], torch.device('cpu'))
    report = permutation_meta(model, frame.iloc[200:], repeats=4, seed=3)
    rows = {row['feature']: row for row in report['rows']}
    assert rows['pred_return']['negative_log_loss_drop'] > .1
    assert rows['pred_return']['negative_log_loss_drop'] > rows['diagnostic_noise_0']['negative_log_loss_drop']
    assert rows['diagnostic_noise_0']['is_noise']
    with pytest.raises(ValueError, match='final-test'):
        permutation_meta(model, frame.assign(prediction_role='final_test'))


def test_grouped_ablations_and_sfi_use_the_exact_purged_populations(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    _write_5m_ohlcv(data / 'raw' / 'ETHUSDT-5m-lighter-fixture.csv', rows=160)
    cfg = _v2_config(data, tmp_path / 'runs')
    cfg['features'] = {'include': ['ohlcv', 'returns']}
    cfg['model']['base']['type'] = 'linear'
    cfg['model']['meta_labeler']['type'] = 'logistic'
    cfg['validation']['method'] = 'exact_span_walk_forward'
    cfg['feature_importance'] = {'enabled': True, 'noise_features': 1, 'base_features': ['log_return'],
                                  'repeats': 1, 'single_feature': True, 'ablation_groups': ['returns']}
    dataset = build_multi_horizon_lighter_dataset(cfg)
    fold = build_validation_splits(dataset, cfg)['folds'][-1]
    model, _ = train_base_model(dataset, fold['train'], cfg, torch.device('cpu'))
    predictions = predict_base_model(model, dataset, fold['validation'], torch.device('cpu'))
    predictions['prediction_role'] = 'base_oof'
    candidates, audit, meta = evaluate_meta_fold(predictions, dataset, fold, cfg, torch.device('cpu'), return_model=True)
    import src.experiments.meta_labeling_mvp as mvp
    calls = []
    original = mvp.train_base_model
    def checked(subset, indices, config, device):
        assert np.array_equal(indices, fold['train'])
        calls.append(subset['feature_columns'])
        return original(subset, indices, config, device)
    monkeypatch.setattr(mvp, 'train_base_model', checked)
    result = fold_importance(model, meta, dataset, fold, candidates, cfg, torch.device('cpu'))
    assert result['evaluation_indices'] == fold['validation'].tolist()
    assert result['split_audit']['overlap_count_after'] == 0
    assert ['log_return'] in calls and ['diagnostic_noise_0'] in calls
    grouped = [row for row in result['ablations'] if row['method'] == 'group_ablation'][0]
    assert 'log_return' not in grouped['retained_columns']
    assert 'close' in grouped['retained_columns']
    summary = summarize_importance([result])
    assert summary['auxiliary_comparisons'] == 3
    assert {row['model'] for row in summary['summary']} >= {'base'}
