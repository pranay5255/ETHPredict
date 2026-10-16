from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.meta_labeling_mvp import (
    add_meta_probabilities, alpha_diagnostics, fit_meta_labeler, meta_threshold_table, run_alpha_backtest,
)


def candidate_fixture(n=120):
    rng = np.random.default_rng(22)
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=.7, size=n) > 0).astype(float)
    return pd.DataFrame({'sample_index': np.arange(n)*4, 'timestamp': pd.date_range('2026-01-01', periods=n, freq='20min'),
        'horizon': 'h', 'is_candidate': True, 'meta_label': y, 'prediction_role': 'base_oof',
        'pred_return': x, 'direction_prob': 1 / (1 + np.exp(-x)), 'realized_vol': .01,
        'expected_edge_bps': 20., 'total_cost_bps': 2., 'horizon_bars': 1, 'side': 1, 'close': 100.,
        'label_net_return': np.where(y, .01, -.01), 'label_gross_return': np.where(y, .0102, -.0098),
        'label_span_start_idx': np.arange(n)*4, 'label_span_end_idx': np.arange(n)*4 + 1})


def meta_config(kind='logistic', method='none'):
    return {'experiment': {'seed': 7}, 'model': {'meta_labeler': {'type': kind, 'hidden_size': 4,
            'n_estimators': 5, 'calibration': {'method': method, 'validation_fraction': .25},
            'selection': {'validation_fraction': .25}}},
            'training': {'epochs': 3, 'batch_size': 16, 'early_stopping': {'enabled': True, 'patience': 1, 'min_delta': 10.}},
            'validation': {'embargo_bars': 1}, 'alpha_backtest': {'meta_threshold': .5, 'position_notional': 100.,
            'meta_threshold_grid': [.2, .4, .6, .8], 'meta_threshold_selection': {'enabled': True, 'metric': 'f1'}}}


@pytest.mark.parametrize('kind', ['constant', 'logistic', 'tree', 'mlp'])
@pytest.mark.parametrize('method', ['none', 'platt', 'isotonic'])
def test_meta_registry_and_calibration_reports(kind, method):
    candidates, config = candidate_fixture(), meta_config(kind, method)
    model = fit_meta_labeler(candidates, config, [], torch.device('cpu'))
    probability, manifest = model.predict_proba(candidates), model.manifest()
    assert len(probability) == len(candidates) and np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert manifest['kind'] == kind
    assert manifest['calibration']['method'] == method
    assert manifest['threshold_selection']['source'] == 'purged_meta_training_holdout_only'
    assert manifest['threshold_selection']['status'] == 'completed'
    assert manifest['decision_threshold'] in [.2, .4, .6, .8]
    if kind in {'logistic', 'tree'}:
        assert manifest['interpretability']['values']
    if kind == 'mlp':
        stopping = manifest['training']['early_stopping']
        assert stopping['selected_epoch'] == 1 and stopping['epochs_run'] == 2
    if method != 'none' and kind != 'constant':
        assert manifest['calibration']['status'] == 'completed'
        assert manifest['calibrator']
        # Chronological calibration and threshold-selection populations are disjoint.
        calibration = manifest['calibration']['evaluation_sample_keys']
        selection = manifest['model_selection']['evaluation_sample_keys']
        assert max(row['sample_index'] for row in calibration) < min(row['sample_index'] for row in selection)


@pytest.mark.parametrize('case,reason', [('empty', 'empty_labeled_candidates'), ('single', 'single_class_training_labels')])
def test_explicit_constant_fallback(case, reason):
    candidates = candidate_fixture()
    candidates['meta_label'] = np.nan if case == 'empty' else 1.
    model = fit_meta_labeler(candidates, meta_config(), [], torch.device('cpu'))
    assert model.manifest()['kind'] == 'constant'
    assert model.manifest()['fallback_reason'] == reason


def test_threshold_tables_monotone_coverage_and_simulated_pnl():
    candidates = candidate_fixture()
    candidates['meta_prob'] = np.linspace(.01, .99, len(candidates))
    rows = meta_threshold_table(candidates, [.2, .4, .6, .8], config=meta_config())
    assert [row['selected'] for row in rows] == sorted([row['selected'] for row in rows], reverse=True)
    assert all({'f1', 'trade_count', 'validation_net_pnl', 'calibration_error'} <= row.keys() for row in rows)
    for row in rows:
        config = meta_config()
        config['alpha_backtest']['meta_threshold'] = row['threshold']
        assert run_alpha_backtest(candidates, config)[0]['net_pnl'] == row['validation_net_pnl']


def test_threshold_and_model_selection_cannot_read_final_test():
    candidates, config = candidate_fixture(), meta_config(method='platt')
    config['model']['meta_labeler']['selection'].update(enabled=True, models=['constant', 'logistic', 'tree'])
    model = fit_meta_labeler(candidates, config, [], torch.device('cpu'))
    saved = deepcopy(model.manifest())
    heldout = candidate_fixture(10)
    heldout['prediction_role'] = 'final_test'
    scored = add_meta_probabilities(heldout, model)
    assert (scored.meta_decision_threshold == model.decision_threshold).all()
    assert alpha_diagnostics(scored, config)['meta_label_thresholds'] == []
    heldout['meta_label'] = 1 - heldout['meta_label']
    add_meta_probabilities(heldout, model)
    assert model.manifest() == saved
    with pytest.raises(ValueError, match='out-of-fold'):
        fit_meta_labeler(heldout, config, [], torch.device('cpu'))
    with pytest.raises(ValueError, match='final-test'):
        meta_threshold_table(scored, config=config)


def test_calibration_labels_do_not_fit_model_or_scaler():
    frame, config = candidate_fixture(), meta_config(method='platt')
    first = fit_meta_labeler(frame, config, [], torch.device('cpu'))
    cal_keys = first.manifest()['calibration']['evaluation_sample_keys']
    changed = frame.copy()
    changed.loc[changed.sample_index.isin([row['sample_index'] for row in cal_keys]), 'meta_label'] = 1 - changed.meta_label
    second = fit_meta_labeler(changed, config, [], torch.device('cpu'))
    assert first.base.manifest() == second.base.manifest()
    assert first.manifest()['calibrator'] != second.manifest()['calibrator']
