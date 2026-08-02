import numpy as np
import pandas as pd

from src.features.sample_weights import (
    average_uniqueness,
    build_label_spans,
    effective_sample_size,
    sample_weights_and_diagnostics,
    span_concurrency,
)


def test_non_overlapping_spans_have_full_uniqueness():
    starts = [0, 2]
    ends = [1, 3]

    concurrency, offset = span_concurrency(starts, ends)
    uniqueness = average_uniqueness(starts, ends)

    assert offset == 0
    assert np.allclose(concurrency, [1.0, 1.0, 1.0, 1.0])
    assert np.allclose(uniqueness, [1.0, 1.0])
    assert effective_sample_size(uniqueness) == 2.0


def test_identical_spans_share_uniqueness_evenly():
    starts = [0, 0, 0]
    ends = [2, 2, 2]

    concurrency, _ = span_concurrency(starts, ends)
    uniqueness = average_uniqueness(starts, ends)

    assert np.allclose(concurrency, [3.0, 3.0, 3.0])
    assert np.allclose(uniqueness, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert np.isclose(effective_sample_size(uniqueness), 1.0)


def test_partially_overlapping_spans_compute_expected_concurrency_and_ess():
    starts = [0, 1]
    ends = [2, 3]

    concurrency, _ = span_concurrency(starts, ends)
    uniqueness = average_uniqueness(starts, ends)

    assert np.allclose(concurrency, [1.0, 2.0, 2.0, 1.0])
    assert np.allclose(uniqueness, [2.0 / 3.0, 2.0 / 3.0])
    assert np.isclose(effective_sample_size(uniqueness), 4.0 / 3.0)


def test_label_spans_prefer_realized_t1_and_emit_timestamps():
    timestamps = pd.date_range("2026-01-01", periods=6, freq="5min")

    spans = build_label_spans([1, 2], [5, 5], timestamps=timestamps, t1_idx=[3, np.nan])

    assert spans["label_span_start_idx"].tolist() == [1, 2]
    assert spans["label_span_end_idx"].tolist() == [3, 5]
    assert spans["label_span_start_timestamp"].iloc[0] == timestamps[1]
    assert spans["label_span_end_timestamp"].iloc[0] == timestamps[3]


def test_weight_modes_normalize_and_handle_empty_or_single_rows():
    frame = pd.DataFrame(
        {
            "label_span_start_idx": [0, 0, 3],
            "label_span_end_idx": [4, 1, 4],
            "t1_idx": [2, 1, 4],
            "label_net_return": [0.01, 0.03, 0.02],
        }
    )

    uniform, uniform_diag = sample_weights_and_diagnostics(frame, mode="uniform")
    horizon, _ = sample_weights_and_diagnostics(frame, mode="horizon_span_uniqueness")
    t1, _ = sample_weights_and_diagnostics(frame, mode="triple_barrier_t1_uniqueness")
    returns, _ = sample_weights_and_diagnostics(frame, mode="return_magnitude")
    combined, combined_diag = sample_weights_and_diagnostics(frame, mode="combined_uniqueness_return")
    empty, empty_diag = sample_weights_and_diagnostics(frame.iloc[0:0], mode="uniform")
    single, single_diag = sample_weights_and_diagnostics(frame.iloc[[0]], mode="horizon_span_uniqueness")

    assert np.allclose(uniform, [1.0 / 3.0] * 3)
    assert not np.allclose(horizon, uniform)
    assert not np.allclose(t1, horizon)
    assert np.isclose(float(returns.sum()), 1.0)
    assert np.allclose(returns.to_numpy(), [1.0 / 6.0, 3.0 / 6.0, 2.0 / 6.0])
    assert np.isclose(float(combined.sum()), 1.0)
    assert combined_diag["weights"]["sum"] == 1.0
    assert uniform_diag["concurrency"]["max"] > 1.0
    assert empty.empty
    assert empty_diag["weights"]["count"] == 0
    assert np.isclose(float(single.iloc[0]), 1.0)
    assert single_diag["effective_sample_size"] == 1.0
