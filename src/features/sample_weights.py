"""Sample span, uniqueness, and weighting helpers for AFML-style labels."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


WEIGHT_MODES = {
    "uniform",
    "horizon_span_uniqueness",
    "triple_barrier_t1_uniqueness",
    "return_magnitude",
    "combined_uniqueness_return",
}


def _numeric_array(values: Any, *, length: Optional[int] = None, default: float = np.nan) -> np.ndarray:
    if values is None:
        if length is None:
            return np.array([], dtype=float)
        return np.full(length, default, dtype=float)
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if length is not None and len(array) != length:
        raise ValueError(f"Expected {length} values, got {len(array)}")
    return array


def _integer_array(values: Any, *, length: Optional[int] = None, default: int = -1) -> np.ndarray:
    array = _numeric_array(values, length=length, default=float(default))
    array = np.where(np.isfinite(array), np.rint(array), default)
    return array.astype(int)


def _timestamp_lookup(timestamps: Optional[Sequence[Any]], indices: np.ndarray) -> pd.Series:
    if timestamps is None:
        return pd.Series([pd.NaT] * len(indices), dtype="datetime64[ns]")
    values = list(timestamps)
    out = []
    for raw_idx in indices:
        idx = int(raw_idx)
        out.append(values[idx] if 0 <= idx < len(values) else pd.NaT)
    return pd.Series(out)


def build_label_spans(
    start_idx: Sequence[Any],
    end_idx: Optional[Sequence[Any]] = None,
    *,
    timestamps: Optional[Sequence[Any]] = None,
    t1_idx: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """Build inclusive label spans, preferring realized ``t1_idx`` when present."""

    starts = _integer_array(start_idx)
    ends = _integer_array(end_idx, length=len(starts), default=-1) if end_idx is not None else starts.copy()
    if t1_idx is not None:
        realized = _numeric_array(t1_idx, length=len(starts))
        ends = np.where(np.isfinite(realized), np.rint(realized), ends).astype(int)
    ends = np.where(ends >= starts, ends, starts).astype(int)
    return pd.DataFrame(
        {
            "label_span_start_idx": starts,
            "label_span_end_idx": ends,
            "label_span_start_timestamp": _timestamp_lookup(timestamps, starts),
            "label_span_end_timestamp": _timestamp_lookup(timestamps, ends),
        }
    )


def span_concurrency(start_idx: Sequence[Any], end_idx: Sequence[Any]) -> Tuple[np.ndarray, int]:
    """Return inclusive concurrency counts and the offset of the first element."""

    starts = _integer_array(start_idx)
    ends = _integer_array(end_idx, length=len(starts), default=-1)
    valid = (starts >= 0) & (ends >= starts)
    if not np.any(valid):
        return np.array([], dtype=float), 0
    starts = starts[valid]
    ends = ends[valid]
    offset = int(starts.min())
    concurrency = np.zeros(int(ends.max()) - offset + 1, dtype=float)
    for left, right in zip(starts, ends):
        concurrency[int(left) - offset : int(right) - offset + 1] += 1.0
    return concurrency, offset


def average_uniqueness(start_idx: Sequence[Any], end_idx: Sequence[Any]) -> np.ndarray:
    """Return per-row average uniqueness, aligned to the input spans."""

    starts = _integer_array(start_idx)
    ends = _integer_array(end_idx, length=len(starts), default=-1)
    concurrency, offset = span_concurrency(starts, ends)
    out = np.zeros(len(starts), dtype=float)
    if len(concurrency) == 0:
        return out
    for row_idx, (left, right) in enumerate(zip(starts, ends)):
        if left < 0 or right < left:
            continue
        active = concurrency[int(left) - offset : int(right) - offset + 1]
        active = active[active > 0]
        out[row_idx] = float(np.mean(1.0 / active)) if len(active) else 0.0
    return out


def effective_sample_size(uniqueness: Sequence[Any]) -> float:
    """AFML effective sample size: the sum of per-row uniqueness values."""

    values = _numeric_array(uniqueness)
    values = values[np.isfinite(values) & (values > 0)]
    return float(values.sum()) if len(values) else 0.0


def normalize_sample_weights(weights: Sequence[Any], normalize: Any = "sum_one") -> np.ndarray:
    """Clean non-negative weights and optionally normalize them."""

    values = _numeric_array(weights)
    if len(values) == 0:
        return values.astype(float)
    values = np.where(np.isfinite(values) & (values >= 0.0), values, 0.0)
    if not np.any(values > 0.0):
        values = np.ones(len(values), dtype=float)

    if normalize is True:
        normalize = "sum_one"
    elif normalize is False or normalize is None:
        normalize = "none"
    mode = str(normalize).lower()
    if mode in {"none", "raw"}:
        return values.astype(float)
    if mode in {"sum_one", "sum", "unit_sum"}:
        return (values / values.sum()).astype(float)
    if mode in {"mean_one", "mean", "unit_mean"}:
        return (values / values.mean()).astype(float)
    raise ValueError(f"Unsupported sample weight normalization: {normalize!r}")


def _return_magnitude(frame: pd.DataFrame, return_values: Optional[Any]) -> np.ndarray:
    if return_values is not None:
        values = np.asarray(return_values, dtype=float)
        if values.ndim == 0:
            values = np.full(len(frame), abs(float(values)), dtype=float)
        elif values.ndim == 1:
            values = np.abs(values)
        else:
            values = np.nanmean(np.abs(values), axis=1)
        if len(values) != len(frame):
            raise ValueError(f"Expected {len(frame)} return values, got {len(values)}")
        return values.astype(float)
    for column in ["label_net_return", "label_gross_return", "true_return", "y_ret"]:
        if column in frame.columns:
            return np.abs(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float))
    return np.ones(len(frame), dtype=float)


def _positive_or_uniform(values: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(values) & (values >= 0.0), values, 0.0)
    return clean if np.any(clean > 0.0) else np.ones(len(clean), dtype=float)


def sample_weights_and_diagnostics(
    frame: pd.DataFrame,
    *,
    mode: str = "uniform",
    normalize: Any = "sum_one",
    return_values: Optional[Any] = None,
    start_col: str = "label_span_start_idx",
    end_col: str = "label_span_end_idx",
    t1_col: str = "t1_idx",
) -> Tuple[pd.Series, dict]:
    """Compute normalized sample weights and the matching span diagnostics."""

    mode = str(mode or "uniform").lower()
    if mode not in WEIGHT_MODES:
        raise ValueError(f"Unsupported sample weight mode: {mode!r}")
    if frame.empty:
        weights = pd.Series([], index=frame.index, dtype=float)
        return weights, sample_weight_diagnostics([], [], [], mode=mode, normalize=normalize)

    if start_col not in frame.columns or end_col not in frame.columns:
        starts = np.arange(len(frame), dtype=int)
        ends = starts.copy()
    else:
        starts = _integer_array(frame[start_col])
        ends = _integer_array(frame[end_col], length=len(starts), default=-1)
    if mode == "triple_barrier_t1_uniqueness" and t1_col in frame.columns:
        t1 = _numeric_array(frame[t1_col], length=len(starts))
        ends = np.where(np.isfinite(t1), np.rint(t1), ends).astype(int)
        ends = np.where(ends >= starts, ends, starts).astype(int)

    uniqueness = average_uniqueness(starts, ends)
    magnitude = _positive_or_uniform(_return_magnitude(frame, return_values))
    if mode == "uniform":
        raw = np.ones(len(frame), dtype=float)
    elif mode in {"horizon_span_uniqueness", "triple_barrier_t1_uniqueness"}:
        raw = uniqueness
    elif mode == "return_magnitude":
        raw = magnitude
    elif mode == "combined_uniqueness_return":
        raw = uniqueness * magnitude
    else:  # pragma: no cover - guarded above
        raw = np.ones(len(frame), dtype=float)

    values = normalize_sample_weights(raw, normalize=normalize)
    weights = pd.Series(values, index=frame.index, dtype=float)
    diagnostics = sample_weight_diagnostics(starts, ends, values, mode=mode, normalize=normalize, uniqueness=uniqueness)
    return weights, diagnostics


def _quantile_summary(values: Sequence[Any], *, include_sum: bool) -> dict:
    clean = _numeric_array(values)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        out = {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        if include_sum:
            out["sum"] = 0.0
        return out
    out = {
        "count": int(len(clean)),
        "min": float(np.min(clean)),
        "p50": float(np.quantile(clean, 0.50)),
        "p95": float(np.quantile(clean, 0.95)),
        "max": float(np.max(clean)),
    }
    if include_sum:
        out["sum"] = float(np.sum(clean))
    return out


def sample_weight_diagnostics(
    start_idx: Sequence[Any],
    end_idx: Sequence[Any],
    weights: Sequence[Any],
    *,
    mode: str,
    normalize: Any,
    uniqueness: Optional[Sequence[Any]] = None,
) -> dict:
    concurrency, _ = span_concurrency(start_idx, end_idx)
    active_concurrency = concurrency[concurrency > 0]
    uniqueness_values = average_uniqueness(start_idx, end_idx) if uniqueness is None else _numeric_array(uniqueness)
    return {
        "mode": str(mode),
        "normalize": "sum_one" if normalize is True else ("none" if normalize is False or normalize is None else str(normalize)),
        "average_uniqueness": float(np.mean(uniqueness_values)) if len(uniqueness_values) else 0.0,
        "effective_sample_size": effective_sample_size(uniqueness_values),
        "concurrency": _quantile_summary(active_concurrency, include_sum=False),
        "weights": _quantile_summary(weights, include_sum=True),
    }


def sample_weight_config(config: Mapping[str, Any]) -> dict:
    raw = dict((config.get("sample_weights", {}) or {}) if isinstance(config, Mapping) else {})
    return {
        "base_mode": str(raw.get("base_mode", "uniform")).lower(),
        "meta_mode": str(raw.get("meta_mode", "uniform")).lower(),
        "normalize": raw.get("normalize", "sum_one"),
    }
