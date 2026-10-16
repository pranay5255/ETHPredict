import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.config.loader import ConfigManager
from src.data.features_all import DataPreprocessor
from src.data.side_data import DataCoverageError, GROUP_INPUTS
from src.experiments.meta_labeling_mvp import build_multi_horizon_lighter_dataset, build_validation_splits, _raw_data_manifest, _run_trial_safe
from tests.test_staged_trial import _v2_config, _write_5m_ohlcv


def fixture(tmp_path, rows=140):
    data = tmp_path / 'data'
    _write_5m_ohlcv(data / 'raw' / 'ETHUSDT-5m-lighter-fixture.csv', rows=rows, step=.002)
    cfg = _v2_config(data, tmp_path / 'runs')
    cfg['validation'].update(method='exact_span_walk_forward')
    cfg['benchmark'] = {'enabled': False}
    cfg['research'] = {'final_test_guard': {'evaluate': False}}
    groups = {}
    time = pd.date_range('2026-03-28', periods=rows, freq='5min')
    values = {'funding_rate': .0001, 'mark_price': 3100., 'open_interest': 5000., 'exchange_volume': 10000.,
              'bid_price': 2999., 'ask_price': 3001., 'bid_depth': 10., 'ask_depth': 5., 'buy_volume': 20., 'sell_volume': 10.}
    for name, columns in GROUP_INPUTS.items():
        frame = pd.DataFrame({'event_time': time, 'published_at': time + pd.Timedelta(minutes=2),
                              **{column: np.full(rows, values[column]) for column in columns}})
        path = data / 'lighter' / f'{name}.csv'; path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        groups[name] = {'paths': [f'lighter/{name}.csv'], 'historical': True, 'timestamp_column': 'event_time',
                        'available_at_column': 'published_at', 'tolerance': '5min', 'min_coverage': 1., 'fill_policy': 'reject'}
    cfg['data']['side_data'] = {'groups': groups}
    return cfg


@pytest.mark.parametrize('bar_type', ['time', 'volume', 'dollar'])
def test_all_historical_groups_reach_active_matrix_and_split_manifests(tmp_path, bar_type):
    cfg = fixture(tmp_path)
    cfg['bars'] = {'type': bar_type, 'threshold_volume': 50., 'threshold_usd': 150_000., 'min_rows': 30,
                   'min_rows_per_split': {'train': 4, 'validation': 4, 'test': 4}}
    dataset = build_multi_horizon_lighter_dataset(cfg)
    assert set(dataset['side_data_manifest']['enabled_groups']) == set(GROUP_INPUTS)
    assert dataset['side_data_manifest']['future_matches'] == 0
    assert dataset['side_data_manifest']['join_clock'] == 'completed_bar_close'
    assert {'funding_rate', 'mark_basis', 'spread_bps', 'trade_flow_imbalance'} <= set(dataset['feature_columns'])
    assert np.isfinite(dataset['X'].numpy()).all()
    splits = build_validation_splits(dataset, cfg)
    assert all(all(value == 1 for value in part['side_data'].values()) for part in splits['data_coverage'].values())
    assert len(dataset['side_data_manifest']['source_files']) == 6
    assert dataset['bar_manifest']['coverage']['5m']['source_coverage']['missing_intervals'] == 0
    if bar_type != 'time':
        assert 'proxy' in dataset['bar_manifest']['coverage']['5m']['construction']


def test_side_feature_family_does_not_replace_price_path(tmp_path):
    cfg = fixture(tmp_path)
    cfg['features'] = {'include': ['returns']}
    dataset = build_multi_horizon_lighter_dataset(cfg)
    assert 'close' not in dataset['feature_columns']
    assert {'open', 'high', 'low', 'close'} <= set(dataset['price_path'])
    trial = _run_trial_safe({'config': cfg, 'trial_id': 'side', 'trial_index': 0}, tmp_path / 'run', smoke=False, device=torch.device('cpu'))
    assert trial['status'] == 'completed', trial.get('reason')
    assert trial['dataset']['side_data_manifest']['status'] == 'validated'
    split = json.loads(Path(trial['split_manifest_path']).read_text())
    assert split['data_coverage']['test']['side_data']['funding'] == 1


def test_publication_time_prevents_future_features(tmp_path):
    cfg = fixture(tmp_path)
    cfg['data']['side_data']['groups'] = {'funding': cfg['data']['side_data']['groups']['funding']}
    group = cfg['data']['side_data']['groups']['funding']
    group.update(fill_policy='zero_with_indicator', min_coverage=.9)
    path = Path(cfg['data']['dir']) / 'lighter' / 'funding.csv'
    frame = pd.read_csv(path)
    frame['funding_rate'] = np.arange(len(frame), dtype=float)
    # Events are recorded at open but published after that candle has closed.
    frame['published_at'] = pd.to_datetime(frame['event_time']) + pd.Timedelta(minutes=6)
    frame.to_csv(path, index=False)
    dataset = build_multi_horizon_lighter_dataset(cfg)
    features = dataset['feature_frame']
    assert features.iloc[0].funding_missing == 1
    assert features.iloc[1].funding_rate == 0
    assert features.iloc[2].funding_rate == 1
    before = features.iloc[:50].copy()
    frame.loc[50:, 'funding_rate'] = 1e9
    frame.to_csv(path, index=False)
    later = build_multi_horizon_lighter_dataset(cfg)['feature_frame']
    pd.testing.assert_frame_equal(before, later.iloc[:50])


def test_global_coverage_cannot_hide_missing_training_split(tmp_path):
    cfg = fixture(tmp_path)
    group = cfg['data']['side_data']['groups']['funding']
    cfg['data']['side_data']['groups'] = {'funding': group}
    group.update(fill_policy='zero_with_indicator', min_coverage=.7, min_split_coverage=.95)
    path = Path(cfg['data']['dir']) / 'lighter' / 'funding.csv'
    pd.read_csv(path).iloc[25:].to_csv(path, index=False)
    dataset = build_multi_horizon_lighter_dataset(cfg)
    with pytest.raises(DataCoverageError, match='split coverage') as error:
        build_validation_splits(dataset, cfg)
    assert 'funding' in error.value.report['side_data']


@pytest.mark.parametrize('failure', ['latest', 'no_availability', 'future_publication', 'sparse', 'conflicting_duplicate', 'unbounded_fill'])
def test_untrustworthy_sources_are_rejected(tmp_path, failure):
    cfg = fixture(tmp_path)
    group = cfg['data']['side_data']['groups']['funding']; cfg['data']['side_data']['groups'] = {'funding': group}
    path = Path(cfg['data']['dir']) / 'lighter' / 'funding.csv'
    frame = pd.read_csv(path)
    if failure == 'latest':
        new = path.with_name('funding_latest.csv'); path.rename(new); group['paths'] = ['lighter/funding_latest.csv']
    elif failure == 'no_availability':
        del group['available_at_column']
    elif failure == 'future_publication':
        frame['published_at'] = pd.to_datetime(frame.event_time) - pd.Timedelta(minutes=1); frame.to_csv(path, index=False)
    elif failure == 'sparse':
        frame.iloc[:1].to_csv(path, index=False)
    elif failure == 'conflicting_duplicate':
        extra = frame.iloc[:1].copy(); extra['funding_rate'] = .1
        pd.concat([frame, extra]).to_csv(path, index=False)
    elif failure == 'unbounded_fill':
        group['fill_policy'] = 'backfill'
    with pytest.raises(ValueError):
        build_multi_horizon_lighter_dataset(cfg)


def test_identical_duplicates_are_audited_and_source_changes_are_hashed(tmp_path):
    cfg = fixture(tmp_path)
    path = Path(cfg['data']['dir']) / 'lighter' / 'funding.csv'
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    dataset = build_multi_horizon_lighter_dataset(cfg)
    assert dataset['side_data_manifest']['coverage']['funding']['duplicate_rows_removed'] == 1
    first = _raw_data_manifest(cfg)
    frame.loc[0, 'funding_rate'] = .1; frame.to_csv(path, index=False)
    assert first != _raw_data_manifest(cfg)


def test_activity_bars_reject_missing_source_intervals_and_conflicting_candles(tmp_path):
    cfg = fixture(tmp_path)
    path = next((Path(cfg['data']['dir']) / 'raw').glob('*.csv'))
    frame = pd.read_csv(path, header=None)
    frame.drop(index=30).to_csv(path, header=False, index=False)
    with pytest.raises(DataCoverageError, match='interval coverage') as error:
        DataPreprocessor(cfg['data']['dir'], bar_config={'type': 'volume', 'threshold_volume': 50}).load_price_data()
    assert error.value.report['missing_intervals'] == 1
    extra = frame.iloc[:1].copy(); extra[4] = extra[4] * .999
    pd.concat([frame, extra]).to_csv(path, header=False, index=False)
    with pytest.raises(DataCoverageError, match='Conflicting duplicate'):
        DataPreprocessor(cfg['data']['dir']).load_price_data()


def test_source_schema_is_validated_and_unknown_groups_fail(tmp_path):
    cfg = fixture(tmp_path)
    path = tmp_path / 'source.yml'; path.write_text(yaml.safe_dump(cfg))
    ConfigManager().load_config(str(path))
    cfg['data']['side_data']['groups']['funding']['fill_policy'] = 'backfill'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='Schema validation'):
        ConfigManager().load_config(str(path))


@pytest.mark.parametrize('bar_type', ['time', 'volume'])
def test_validation_only_pipeline_connects_all_three_issues(tmp_path, monkeypatch, bar_type):
    import src.experiments.meta_labeling_mvp as pipeline
    cfg = fixture(tmp_path)
    cfg['bars'] = {'type': bar_type, 'threshold_volume': 50., 'min_rows': 30}
    cfg['labels']['meta_triple_barrier'].update(success_mode='profit_take_only', intrabar_ordering='exclude')
    cfg['tracking'] = {'trackio': {'enabled': False, 'allow_local_debug_without_trackio': True}}
    cfg['search'] = {'mode': 'grid', 'max_trials': 2, 'trials': [
        {'id': 'baseline', 'overrides': {}},
        {'id': 'symmetric', 'overrides': {'labels.meta_triple_barrier.geometry': 'symmetric'}},
    ]}
    cfg['smoke']['max_trials'] = 2
    config_path = tmp_path / 'integration.yml'; config_path.write_text(yaml.safe_dump(cfg))
    def reject_test_query(*args, **kwargs):
        raise AssertionError('Implementation fixture must not query the final test')
    monkeypatch.setattr(pipeline, '_record_final_test_evaluation', reject_test_query)
    result = pipeline.run_meta_labeling_mvp(config_path, smoke=True, device='cpu', allow_cpu=True)
    assert len(result['trials']) == 2
    assert result['trial_accounting']['trial_count'] == 2
    assert Path(result['stage0']['bars_path']).is_file()
    assert pd.read_parquet(result['stage0']['bars_path']).timestamp.is_monotonic_increasing
    for trial in result['trials']:
        assert trial['status'] == 'completed', trial.get('reason')
        assert trial['test_evaluation']['status'] == 'disabled'
        assert trial['metrics']['test'] == {}
        assert trial['final_test_evaluation'] is None
        assert trial['dataset']['side_data_manifest']['status'] == 'validated'
        candidates = pd.read_parquet(trial['artifact_paths']['oof_candidates'])
        assert set(candidates.prediction_role) == {'base_oof'}
        assert 'held_out' in set(candidates.meta_evaluation_status)
        assert {'ambiguity_flag', 'label_success_mode', 'profit_barrier_price'} <= set(candidates)
        split = json.loads(Path(trial['split_manifest_path']).read_text())
        assert split['final_boundary_audit']['overlap_count_after'] == 0
        assert all(part['side_data']['funding'] == 1 for part in split['data_coverage'].values())
        assert json.loads(Path(trial['trackio_receipt_path']).read_text())['delivery'] == 'disabled'
        if bar_type != 'time':
            assert trial['metrics']['validation']['statistics']['periods_per_year'] is None
    assert not list(Path(result['run_dir']).rglob('*test*.parquet'))


def test_disabled_side_sources_preserve_baseline_matrix(tmp_path):
    cfg = fixture(tmp_path)
    enabled = build_multi_horizon_lighter_dataset(cfg)
    cfg['data']['side_data'] = {'groups': {}}
    baseline = build_multi_horizon_lighter_dataset(cfg)
    count = baseline['input_size']
    assert torch.equal(enabled['X'][:, :, :count], baseline['X'])
    assert torch.equal(enabled['y_ret'], baseline['y_ret'])
    assert baseline['side_data_manifest']['status'] == 'disabled'


def test_group_order_is_deterministic_and_incomplete_activity_bar_is_excluded(tmp_path):
    cfg = fixture(tmp_path)
    first = build_multi_horizon_lighter_dataset(cfg)
    cfg['data']['side_data']['groups'] = dict(reversed(list(cfg['data']['side_data']['groups'].items())))
    other = build_multi_horizon_lighter_dataset(cfg)
    assert first['feature_columns'] == other['feature_columns']
    assert torch.equal(first['X'], other['X'])
    data = tmp_path / 'terminal'
    _write_5m_ohlcv(data / 'raw' / 'ETHUSDT-5m-lighter-fixture.csv', rows=20)
    preprocessor = DataPreprocessor(str(data), bar_config={'type': 'volume', 'threshold_volume': 100})
    bars = preprocessor.load_price_data()['5m']
    report = preprocessor.bar_manifest()['coverage']['5m']
    assert len(bars) == 3
    assert report['incomplete_terminal_bar_excluded']
    assert report['excluded_terminal_source_rows'] == 2
    assert report['retained_source_rows'] == 18


def test_nonzero_static_hourly_funding_is_rejected_on_activity_clock(tmp_path):
    cfg = fixture(tmp_path)
    cfg['bars'] = {'type': 'volume', 'threshold_volume': 50}
    cfg['costs']['funding_bps_per_hour'] = 1.
    with pytest.raises(ValueError, match='elapsed-time funding'):
        build_multi_horizon_lighter_dataset(cfg)


def test_data_bar_type_override_controls_the_active_clock(tmp_path):
    cfg = fixture(tmp_path)
    cfg['bars'] = {'type': 'time', 'threshold_volume': 50.}
    cfg['data']['bar_type'] = 'volume'
    dataset = build_multi_horizon_lighter_dataset(cfg)
    assert dataset['bar_manifest']['coverage']['5m']['bar_type'] == 'volume'
    assert len(dataset['price_path']) < 140


def test_exact_missing_interval_budget_is_not_rejected_by_float_cancellation():
    from src.data.side_data import validate_split_coverage
    time = pd.date_range('2026-01-01', periods=101, freq='5min').delete(50)
    opening = time.astype('datetime64[ms]').asi8
    dataset = {'samples': pd.DataFrame({'sample_index': [99]}), 'sequence_length': 99,
               'price_path': pd.DataFrame({'open_time': opening, 'close_time': opening + 299999}),
               'source_timestamps': time,
               'coverage_config': {'source_interval_ms': 300000, 'bars': {'max_missing_fraction': .01}}}
    reports = validate_split_coverage(dataset, {'development': np.array([0]), 'test': np.array([0]), 'folds': []})
    assert reports['development']['missing_source_intervals'] == 1
    assert reports['development']['source_interval_coverage'] == .99
