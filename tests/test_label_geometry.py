import numpy as np
import pandas as pd
import pytest
import torch

from src.features.labeling import meta_triple_barrier_labels
from src.experiments.meta_labeling_mvp import fit_meta_labeler, label_diagnostics


def path(high=100.5, low=99.5, close=100.3):
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="5min"),
                         "open": [100., 100., 100.], "high": [100., high, 100.5],
                         "low": [100., low, 99.5], "close": [100., close, 100.3]})


def label(prices, **config):
    signals = pd.DataFrame([{"sample_index": 0, "horizon_bars": 2, "side": 1,
                             "realized_vol": .01, "is_candidate": True, "total_cost_bps": 1.}])
    return meta_triple_barrier_labels(signals, prices, {"profit_kappa": 1., "stop_kappa": 1., **config}, {}).iloc[0]


@pytest.mark.parametrize("success, expected", [("positive_net_at_vertical", 1), ("profit_take_only", 0)])
def test_expiry_success_is_an_explicit_definition(success, expected):
    result = label(path(), success_mode=success)
    assert result.meta_label == expected
    assert result.exit_reason == "vertical"
    assert result.label_status == "valid"
    assert result.first_touch_timestamp == result.vertical_barrier_timestamp
    assert result.entry_price == 100
    assert result.label_net_return == pytest.approx(np.log(100.3 / 100) - .0001)


@pytest.mark.parametrize("side, high, low, expected", [(1, 102, 99.5, "profit_take"), (1, 100.5, 98, "stop_loss"),
                                                       (-1, 100.5, 98, "profit_take"), (-1, 102, 99.5, "stop_loss")])
def test_long_short_first_touch(side, high, low, expected):
    signals = pd.DataFrame([{"sample_index": 0, "horizon_bars": 2, "side": side, "realized_vol": .01, "is_candidate": True}])
    result = meta_triple_barrier_labels(signals, path(high, low, 100), {}, {}).iloc[0]
    assert result.exit_reason == expected
    assert result.meta_label == float(expected == "profit_take")
    assert result.t1_idx == 1
    assert result.label_span_end_timestamp == result.first_touch_timestamp
    assert result.profit_barrier_price == pytest.approx(100 * np.exp(side * result.profit_barrier))


@pytest.mark.parametrize("ordering, expected", [("stop_first", 0), ("profit_first", 1), ("exclude", None)])
def test_ambiguous_ohlc_is_never_silent(ordering, expected):
    result = label(path(103, 97, 100), intrabar_ordering=ordering)
    assert result.ambiguity_flag
    if expected is None:
        assert pd.isna(result.meta_label)
        assert result.label_exclusion_reason == "ambiguous"
    else:
        assert result.meta_label == expected
        assert result.label_status == "valid"


def test_symmetric_geometry_uses_equal_cost_adjusted_widths():
    result = label(path(), geometry="symmetric", stop_kappa=2)
    assert result.profit_barrier == result.stop_barrier == .02


def test_missing_prices_are_not_backfilled_and_unfinished_paths_are_not_failures():
    prices = path()
    prices.loc[1, "high"] = np.nan
    result = label(prices)
    assert pd.isna(result.meta_label)
    assert result.label_exclusion_reason == "invalid_ohlc"
    result = label(path().iloc[:2])
    assert pd.isna(result.meta_label)
    assert result.label_exclusion_reason == "incomplete_path"
    assert result.vertical_barrier_idx == 2
    assert pd.isna(result.vertical_barrier_timestamp)
    # An observed touch completes a label even before planned expiry exists.
    result = label(path(103, 99.5, 100).iloc[:2])
    assert result.meta_label == 1
    assert result.t1_idx == 1


@pytest.mark.parametrize("config", [{"profit_kappa": 0}, {"stop_kappa": np.nan}, {"success_mode": "typo"},
                                    {"intrabar_ordering": "typo"}, {"geometry": "typo"}])
def test_invalid_definitions_fail(config):
    with pytest.raises(ValueError):
        label(path(), **config)


def test_non_candidates_and_unavailable_labels_never_train_meta_model():
    frame = pd.DataFrame([{"is_candidate": False, "meta_label": 1., "pred_return": 1.},
                          {"is_candidate": True, "meta_label": np.nan, "pred_return": 1.}])
    model = fit_meta_labeler(frame, {}, [], torch.device("cpu"))
    assert model.probability == 0.
    frame.loc[0, "is_candidate"] = True
    frame["prediction_role"] = "final_test"
    with pytest.raises(ValueError, match="out-of-fold"):
        fit_meta_labeler(frame, {}, [], torch.device("cpu"))


def test_diagnostics_include_classification_ambiguity_and_exclusions():
    frame = pd.DataFrame({"is_candidate": [True, True, False], "meta_label": [1., 0., np.nan],
                          "meta_prob": [.9, .1, 0.], "ambiguity_flag": [True, False, False],
                          "expected_edge_bps": [20., 10., -5.], "label_exclusion_reason": [None, None, "no_signal"]})
    report = label_diagnostics(frame, {})
    assert report["ambiguous_rows"] == 1
    assert report["classification"]["f1"] == 1
    assert report["classification"]["calibration_error"] == pytest.approx(.1)
    assert report["exclusion_reasons"] == {"no_signal": 1}


def test_candidate_artifacts_persist_label_evidence(tmp_path):
    from tests.test_staged_trial import _v2_config, _write_5m_ohlcv
    from src.experiments.meta_labeling_mvp import _run_trial_safe
    data = tmp_path / 'data'
    _write_5m_ohlcv(data / 'raw' / 'ETHUSDT-5m-lighter-fixture.csv', rows=96)
    config = _v2_config(data, tmp_path / 'runs')
    config['labels']['meta_triple_barrier'].update(success_mode='profit_take_only', intrabar_ordering='exclude')
    manifest = _run_trial_safe({'config': config, 'trial_id': 'geometry', 'trial_index': 0}, tmp_path / 'runs', smoke=False, device=torch.device('cpu'))
    assert manifest['status'] == 'completed', manifest.get('reason')
    candidates = pd.read_parquet(manifest['artifact_paths']['oof_candidates'])
    assert {'entry_price', 'entry_timestamp', 'first_touch_timestamp', 'profit_barrier_price',
            'label_config_hash', 'ambiguity_flag', 'prediction_role', 'label_success_mode'}.issubset(candidates)
    assert set(candidates.label_success_mode) == {'profit_take_only'}
    assert 'meta_classification' in manifest['metrics']['validation']
