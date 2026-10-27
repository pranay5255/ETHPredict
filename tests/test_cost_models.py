from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.data.side_data import DataCoverageError
from src.features.labeling import meta_triple_barrier_labels
from src.simulator.costs import CostModel
from src.experiments.meta_labeling_mvp import run_alpha_backtest
from src.evaluation.cost_stress import cost_stress_report


def path_fixture():
    return pd.DataFrame({'timestamp': pd.date_range('2026-01-01', periods=8, freq='h'),
        'open': [100., 102., 104., 106., 108., 110., 112., 114.],
        'close': [100., 102., 104., 106., 108., 110., 112., 114.],
        'high': [100.1, 102.1, 104.1, 106.1, 108.1, 110.1, 112.1, 114.1],
        'low': [99.9, 101.9, 103.9, 105.9, 107.9, 109.9, 111.9, 113.9]})


def signal_fixture():
    return {'sample_index': 1, 'timestamp': pd.Timestamp('2026-01-01T01:00:00'), 'horizon': 'h', 'horizon_bars': 2,
            'side': 1, 'is_candidate': True, 'realized_vol': .1, 'close': 102., 'meta_prob': .9, 'expected_edge_bps': 200.}


def test_static_stress_and_label_pnl_share_realized_cost_components():
    path, signal = path_fixture(), signal_fixture()
    cfg = {'fee_bps': 2., 'spread_bps': 3., 'slippage_bps': 1., 'funding_bps_per_hour': 2.}
    model = CostModel(cfg, path, granularity='1h')
    assert model.quote(signal)['total_cost_bps'] == 12.
    stress = CostModel({**cfg, 'mode': 'stress', 'stress_multiplier': 2.}, path, granularity='1h')
    assert stress.quote(signal)['total_cost_bps'] == 24.
    labels = meta_triple_barrier_labels(pd.DataFrame([signal]), path, {}, cfg, cost_model=model, granularity='1h')
    labels['prediction_role'] = 'base_oof'
    metrics, trades = run_alpha_backtest(labels, {'costs': cfg, 'alpha_backtest': {'position_notional': 1000.}})
    row = labels.iloc[0]
    assert trades.iloc[0].net_pnl == pytest.approx(1000*row.label_net_return)
    assert trades.iloc[0].fees == pytest.approx(1000*row.realized_total_cost_bps/10000.)
    assert sum(metrics['cost_components'].values()) == pytest.approx(metrics['fees'])
    assert metrics['gross_pnl'] - metrics['fees'] == pytest.approx(metrics['net_pnl'])
    report = cost_stress_report(labels, {'costs': cfg, 'alpha_backtest': {'position_notional': 1000.}})
    assert report['labels_refit'] is False
    assert report['rows'][0]['net_pnl'] > report['rows'][1]['net_pnl'] > report['rows'][2]['net_pnl']


def test_dynamic_funding_is_signed_and_estimate_cannot_read_future_payments(tmp_path):
    path = path_fixture()
    funding = pd.DataFrame({'timestamp': path.timestamp, 'rate': [1., 2., 3., 4., 5., 6., 7., 8.]})
    source = tmp_path / 'funding.csv'
    funding.to_csv(source, index=False)
    config = {'mode': 'dynamic', 'dynamic_funding': {'enabled': True, 'historical': True,
              'paths': [str(source)], 'timestamp_column': 'timestamp', 'rate_column': 'rate', 'rate_unit': 'bps'}}
    model = CostModel(config, path, granularity='1h')
    signal = signal_fixture()
    assert model.quote(signal)['funding_bps'] == 4.  # Last known 2 bps x 2 hours.
    assert model.quote(signal, exit_index=3)['funding_bps'] == 7.  # Actual payments 3 + 4.
    assert model.quote({**signal, 'side': -1}, exit_index=3)['funding_bps'] == -7.
    funding.loc[funding.index > 1, 'rate'] = 999.
    funding.to_csv(source, index=False)
    changed = CostModel(config, path, granularity='1h')
    assert changed.quote(signal)['funding_bps'] == model.quote(signal)['funding_bps']
    assert changed.quote(signal, exit_index=3)['funding_bps'] != model.quote(signal, exit_index=3)['funding_bps']
    assert changed.identity != model.identity


def test_lighter_percent_and_payer_direction_are_explicit(tmp_path):
    path = path_fixture()
    source = tmp_path / 'funding.csv'
    pd.DataFrame({'timestamp': path.timestamp, 'rate': .001, 'direction': ['long', 'short']*4}).to_csv(source, index=False)
    cfg = {'mode': 'dynamic', 'dynamic_funding': {'enabled': True, 'historical': True, 'paths': [str(source)],
           'rate_column': 'rate', 'rate_unit': 'percent', 'direction_column': 'direction', 'direction_semantics': 'payer'}}
    model = CostModel(cfg, path, granularity='1h')
    assert model.funding.rate_bps.to_list() == pytest.approx([.1, -.1]*4)


def test_delayed_entry_maker_haircut_and_missed_fills():
    path, signal = path_fixture(), signal_fixture()
    costs = {'fee_bps': 2., 'execution': {'mode': 'maker', 'maker_fee_bps': 1., 'entry_delay_bars': 1, 'maker_fill_probability': .5}}
    model = CostModel(costs, path)
    labels = meta_triple_barrier_labels(pd.DataFrame([signal]), path, {}, costs, cost_model=model)
    row = labels.iloc[0]
    assert row.entry_idx == 2 and row.entry_price == 104.
    assert row.label_span_start_idx == 1 and row.vertical_barrier_idx == 4
    assert row.fill_factor == .5
    assert row.label_net_return == pytest.approx(.5*(row.event_gross_return - row.realized_total_cost_bps/10000.))
    costs['execution'].update(fill_policy='sampled', missed_fill_probability=1.)
    missing = meta_triple_barrier_labels(pd.DataFrame([signal]), path, {}, costs, cost_model=CostModel(costs, path))
    assert missing.iloc[0].label_status == 'not_filled' and np.isnan(missing.iloc[0].meta_label)
    assert run_alpha_backtest(missing, {'costs': costs})[0]['trades'] == 0


def test_spread_proxy_depth_slippage_and_historical_coverage_fail_closed(tmp_path):
    path, signal = path_fixture(), signal_fixture()
    cfg = {'mode': 'dynamic', 'dynamic_spread': {'enabled': True, 'source': 'high_low_proxy', 'range_fraction': .1},
           'dynamic_slippage': {'enabled': True, 'volatility_coefficient': .01}}
    model = CostModel(cfg, path)
    quote = model.quote(signal)
    assert quote['spread_bps'] > 0 and quote['slippage_bps'] == pytest.approx(10.)
    cfg['dynamic_spread']['source'] = 'order_book'
    with pytest.raises(DataCoverageError):
        CostModel(cfg, path).quote(signal)
    source = tmp_path / 'funding.csv'
    pd.DataFrame({'timestamp': path.timestamp.iloc[[0, 7]], 'rate': [1., 1.]}).to_csv(source, index=False)
    cfg = {'mode': 'dynamic', 'dynamic_funding': {'enabled': True, 'historical': True, 'paths': [str(source)], 'rate_column': 'rate', 'rate_unit': 'bps'}}
    with pytest.raises(DataCoverageError, match='gap'):
        CostModel(cfg, path)


def test_backtest_rejects_cost_changes_without_regenerated_labels():
    path, signal = path_fixture(), signal_fixture()
    cfg = {'fee_bps': 2.}
    model = CostModel(cfg, path)
    labels = meta_triple_barrier_labels(pd.DataFrame([signal]), path, {}, cfg, cost_model=model)
    with pytest.raises(ValueError, match='regenerate labels'):
        run_alpha_backtest(labels, {'costs': {'fee_bps': 4.}})
    run_alpha_backtest(labels, {'costs': cfg}, cost_model=model)


def test_funding_cashflow_sources_participate_in_raw_data_identity(tmp_path):
    from src.experiments.meta_labeling_mvp import _raw_data_manifest, raw_data_fingerprint
    source = tmp_path / 'funding.csv'
    source.write_text('timestamp,rate\n2026-01-01,1\n')
    cfg = {'data': {'dir': str(tmp_path)}, 'costs': {'mode': 'dynamic', 'dynamic_funding': {'enabled': True, 'paths': [str(source)]}}}
    first = raw_data_fingerprint(_raw_data_manifest(cfg))
    source.write_text('timestamp,rate\n2026-01-01,2\n')
    assert first != raw_data_fingerprint(_raw_data_manifest(cfg))


def test_size_dependent_costs_use_executable_capped_notional_throughout():
    from src.experiments.meta_labeling_mvp import candidate_signals

    path, signal = path_fixture(), signal_fixture()
    cfg = {'costs': {'mode': 'dynamic', 'dynamic_slippage': {'enabled': True,
            'size_coefficient_bps': 2., 'reference_notional': 1000.}},
           'alpha_backtest': {'position_notional': 10000., 'max_position_notional': 1000.}}
    predictions = pd.DataFrame([{**signal, 'h_pred_return': .04, 'h_direction_prob': .9,
                                 'h_true_return': .04, 'h_true_direction': 1.}])
    dataset = {'price_path': path, 'horizon_names': ['h'], 'horizon_bars': {'h': 2}, 'granularity': '1h'}
    candidates = candidate_signals(predictions, dataset, cfg)
    assert candidates.iloc[0].position_notional == 1000.
    assert candidates.iloc[0].slippage_bps == pytest.approx(2.)
    labels = meta_triple_barrier_labels(candidates, path, {}, cfg['costs'],
                                        cost_model=dataset['cost_model'], granularity='1h')
    assert labels.iloc[0].realized_slippage_bps == pytest.approx(2.)
    metrics, trades = run_alpha_backtest(labels, cfg, cost_model=dataset['cost_model'])
    assert trades.iloc[0].notional == 1000.
    assert metrics['cost_components']['slippage'] == pytest.approx(.2)
