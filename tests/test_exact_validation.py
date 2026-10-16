import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.validation.splits import embargo_width, interval_coordinates, overlaps, purge_training_spans
from src.experiments.meta_labeling_mvp import build_multi_horizon_lighter_dataset, build_validation_splits, _run_one_trial
from tests.test_staged_trial import _v2_config, _write_5m_ohlcv


def fixture(tmp_path, method='exact_span_walk_forward'):
    data = tmp_path / 'data'
    _write_5m_ohlcv(data / 'raw' / 'ETHUSDT-5m-lighter-fixture.csv', rows=120, step=.002)
    cfg = _v2_config(data, tmp_path / 'runs')
    cfg['validation'].update(method=method, folds=3, embargo_bars=1, scenario_test_groups=1)
    cfg['research'] = {'final_test_guard': {'evaluate': False}}
    cfg['benchmark'] = {'enabled': False}
    return cfg, build_multi_horizon_lighter_dataset(cfg)


def test_inclusive_irregular_timestamp_overlap_and_realized_t1():
    time = pd.to_datetime(['2026-01-01 10:00', '2026-01-01 10:05', '2026-01-01 10:20', '2026-01-01 10:45'])
    frame = pd.DataFrame({'label_span_start_idx': [0, 1, 2, 3], 'label_span_end_idx': [2, 1, 2, 3],
                          'label_span_start_timestamp': time, 'label_span_end_timestamp': time[[2, 1, 2, 3]]})
    train, audit = purge_training_spans(frame, [0, 1, 3], [2], {'embargo_bars': 1}, clock=time)
    assert train.tolist() == [1]  # realized t1=10:05 survives; endpoint 10:20 does not
    assert audit['purged_indices'] == [0]
    assert audit['embargoed_indices'] == [3]
    assert audit['overlap_count_after'] == 0


@pytest.mark.parametrize('mode', ['fixed_bars', 'max_horizon', 'sequence_plus_horizon', 'realized_t1', 'volatility_adaptive'])
def test_all_embargo_modes_use_source_clock_and_train_local_volatility(mode):
    frame = pd.DataFrame({'label_span_start_idx': [0, 5, 10], 'label_span_end_idx': [2, 7, 12], 'realized_vol': [.01, .02, .99]})
    cfg = {'embargo_mode': mode, 'embargo_bars': 2}
    width = embargo_width(frame, [2], cfg, max_horizon=3, sequence_length=4, train=[0, 1])
    changed = frame.copy(); changed.loc[2, 'realized_vol'] = 1000
    assert width == embargo_width(changed, [2], cfg, max_horizon=3, sequence_length=4, train=[0, 1])
    assert width >= 0


@pytest.mark.parametrize('method', ['purged_walk_forward', 'exact_span_walk_forward', 'purged_kfold', 'grouped_scenario'])
def test_all_active_splits_protect_every_horizon_and_final_test(tmp_path, method):
    cfg, dataset = fixture(tmp_path, method)
    splits = build_validation_splits(dataset, cfg)
    starts, ends = interval_coordinates(dataset['samples'])
    test = splits['test']; dev = splits['development']
    assert not overlaps(starts[dev], ends[dev], starts[test], ends[test]).any()
    for fold in splits['folds']:
        train, validation = fold['train'], fold['validation']
        assert not overlaps(starts[train], ends[train], starts[validation], ends[validation]).any()
        assert fold['audit']['overlap_count_after'] == 0
    if method == 'grouped_scenario':
        assert len(splits['folds']) == 3


def test_undersized_and_unknown_partitions_fail(tmp_path):
    cfg, dataset = fixture(tmp_path)
    cfg['validation']['min_train_rows'] = 1000
    with pytest.raises(ValueError, match='coverage'):
        build_validation_splits(dataset, cfg)
    cfg['validation'] = {'method': 'typo'}
    with pytest.raises(ValueError, match='Unknown'):
        build_validation_splits(dataset, cfg)


@pytest.mark.parametrize('method', ['exact_span_walk_forward', 'grouped_scenario'])
def test_nested_meta_validation_and_path_artifacts(tmp_path, method):
    cfg, dataset = fixture(tmp_path, method)
    if method == 'grouped_scenario':
        cfg['validation'].update(folds=4, scenario_test_groups=2)
    result = _run_one_trial({'trial_id': 'split', 'trial_index': 0, 'config': cfg}, tmp_path / 'runs', smoke=False, device=torch.device('cpu'))
    candidates = pd.read_parquet(result['artifact_paths']['oof_candidates'])
    assert 'held_out' in set(candidates.meta_evaluation_status)
    for fold in result['folds']:
        if fold['meta_validation'].get('realized_t1_purge'):
            assert fold['meta_validation']['realized_t1_purge']['span_kind'] == 'meta_realized_t1'
            assert fold['meta_validation']['realized_t1_purge']['overlap_count_after'] == 0
        outer_validation = build_validation_splits(dataset, cfg)['folds'][fold['fold']]['validation']
        for inner in fold['meta_validation']['inner_folds']:
            assert not set(inner['fit_indices']) & set(outer_validation)
            assert not set(inner['evaluation_indices']) & set(outer_validation)
            assert inner['overlap_count_after'] == 0
    split = json.loads(open(result['split_manifest_path']).read())
    assert split['final_boundary_audit']['overlap_count_after'] == 0
    assert result['metrics']['test'] == {}
    assert result['metrics']['validation']['path_dispersion']['scenario_count'] == (6 if method == 'grouped_scenario' else 3)
    assert not result['metrics']['validation']['path_dispersion']['stitched_cpcv']


def test_outer_evaluation_outcomes_cannot_change_meta_training(tmp_path):
    from src.experiments.meta_labeling_mvp import fit_outer_meta_model
    cfg, dataset = fixture(tmp_path)
    fold = build_validation_splits(dataset, cfg)['folds'][0]
    changed = dict(dataset)
    changed['y_ret'] = dataset['y_ret'].clone(); changed['y_dir'] = dataset['y_dir'].clone()
    changed['y_ret'][fold['validation']] = 999
    changed['y_dir'][fold['validation']] = 1 - changed['y_dir'][fold['validation']]
    changed['price_path'] = dataset['price_path'].copy()
    first = int(dataset['samples'].iloc[fold['validation'][0]].sample_index)
    changed['price_path'].loc[first:, ['open', 'high', 'low', 'close']] *= 10
    torch.manual_seed(7); model, audit = fit_outer_meta_model(dataset, fold['train'], cfg, torch.device('cpu'))
    torch.manual_seed(7); other, other_audit = fit_outer_meta_model(changed, fold['train'], cfg, torch.device('cpu'))
    assert audit == other_audit
    assert model.manifest() == other.manifest()
    if hasattr(model, 'model'):
        for name, value in model.model.state_dict().items():
            assert torch.equal(value, other.model.state_dict()[name])


def test_purge_normalizes_timestamp_resolutions():
    # pandas preserves input precision; milliseconds and microseconds are not
    # comparable as raw integer representations until both normalize to ns.
    frame = pd.DataFrame({'label_span_start_idx': [0, 1], 'label_span_end_idx': [1, 1]})
    frame['label_span_start_timestamp'] = pd.Series(pd.date_range('2026-01-01', periods=2, freq='5min')).astype('datetime64[ms]')
    frame['label_span_end_timestamp'] = pd.Series([pd.Timestamp('2026-01-01 00:05')] * 2).astype('datetime64[us]')
    retained, audit = purge_training_spans(frame, [0], [1])
    assert len(retained) == 0
    assert audit['purged_indices'] == [0]


def test_scenario_summary_uses_predeclared_order_not_sort_tie_order():
    from src.experiments.meta_labeling_mvp import collapse_scenario_predictions
    frame = pd.DataFrame({'sample_index': [0, 0], 'horizon': ['h', 'h'], 'meta_fold': [2, 0],
                          'meta_prob': [.9, .2], 'timestamp': [pd.Timestamp('2026-01-01')] * 2})
    assert collapse_scenario_predictions(frame, candidates=True).iloc[0].meta_prob == .2
