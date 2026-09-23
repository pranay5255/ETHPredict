"""Single-path backtest statistics and a CSCV probability of backtest overfitting.

Annualised Sharpe uses 5-minute bars on a 24/7 clock: a mean year has 365.25 days
and each day has 288 five-minute bars, so the factor is ``sqrt(105_192)``. This is
not the 252-day equity-market factor.

Probabilistic Sharpe uses the Bailey and López de Prado normal CDF with sample
skewness and non-excess kurtosis (a normal distribution has kurtosis 3), so the
denominator term is ``(kurtosis - 1) / 4``. Deflated Sharpe evaluates that PSR at
the Euler-Mascheroni expected-maximum Sharpe. The null mean is zero. The hurdle
scales with ``sqrt(variance of trial Sharpes)`` and the recorded trial count.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.features.sample_weights import average_uniqueness, effective_sample_size


# 365.25 days * 24 hours * 12 five-minute bars per hour.
FIVE_MINUTE_BARS_PER_YEAR = 365.25 * 24.0 * 12.0
EULER_MASCHERONI = 0.5772156649015329
ANNUALISATION_NOTE = (
    "5-minute bars, 24/7, 365.25-day year "
    f"({int(FIVE_MINUTE_BARS_PER_YEAR)} bars/year). Not a 252-day equity factor."
)


def _finite(values: Sequence[Any]) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _unavailable(reason: str, *, path_type: str = "single_path") -> Dict[str, Any]:
    return {"available": False, "value": "unavailable", "reason": reason, "path_type": path_type}


def _available(value: float, *, path_type: str = "single_path", **extra: Any) -> Dict[str, Any]:
    return {"available": True, "value": float(value), "reason": None, "path_type": path_type, **extra}


def periods_per_year(granularity: str = "5m") -> float:
    """Return the 24/7 annualisation factor for a bar size. Only 5-minute bars are defined."""

    if str(granularity) != "5m":
        raise ValueError(f"Annualisation is documented for 5-minute 24/7 bars, got {granularity!r}")
    return float(FIVE_MINUTE_BARS_PER_YEAR)


def sharpe_ratio(returns: Sequence[Any], *, granularity: str = "5m") -> Dict[str, Any]:
    """Per-period Sharpe and its 5-minute 24/7 annualisation."""

    clean = _finite(returns)
    if str(granularity) != "5m":
        factor = None
        note = {
            "periods_per_year": None,
            "annualisation": "annualisation is documented only for 5-minute 24/7 bars",
            "granularity": str(granularity),
        }
    else:
        factor = periods_per_year(granularity)
        note = {"periods_per_year": factor, "annualisation": ANNUALISATION_NOTE, "granularity": granularity}
    if len(clean) < 2:
        return {
            "per_period": _unavailable("fewer than 2 finite period returns"),
            "annualised": _unavailable("fewer than 2 finite period returns"),
            **note,
        }
    std = float(clean.std(ddof=1))
    if std <= 0.0 or not math.isfinite(std):
        return {
            "per_period": _unavailable("period-return standard deviation is zero"),
            "annualised": _unavailable("period-return standard deviation is zero"),
            **note,
        }
    per_period = float(clean.mean() / std)
    if factor is None:
        annualised = _unavailable("annualisation is documented only for 5-minute 24/7 bars")
    else:
        annualised = _available(float(per_period * math.sqrt(factor)), observations=int(len(clean)))
    return {
        "per_period": _available(per_period, observations=int(len(clean))),
        "annualised": annualised,
        **note,
    }


def _moment_pair(clean: np.ndarray) -> tuple[float, float, float, float]:
    sharpe = float(clean.mean() / clean.std(ddof=1))
    skew = float(stats.skew(clean, bias=False))
    kurtosis = float(stats.kurtosis(clean, fisher=False, bias=False))
    return sharpe, skew, kurtosis, float(len(clean))


def probabilistic_sharpe_ratio(returns: Sequence[Any], *, benchmark_sharpe: float = 0.0) -> Dict[str, Any]:
    """PSR: normal CDF of the Sharpe against ``benchmark_sharpe``.

    Kurtosis is non-excess. A normal sample uses ``(3 - 1) / 4`` in the variance term.
    """

    clean = _finite(returns)
    if len(clean) < 2:
        return _unavailable("fewer than 2 finite period returns")
    std = float(clean.std(ddof=1))
    if std <= 0.0 or not math.isfinite(std):
        return _unavailable("period-return standard deviation is zero")
    sharpe, skew, kurtosis, observations = _moment_pair(clean)
    if not all(math.isfinite(item) for item in (sharpe, skew, kurtosis, benchmark_sharpe)):
        return _unavailable("Sharpe moments are not finite")
    variance_term = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    if variance_term <= 0.0 or not math.isfinite(variance_term):
        return _unavailable("PSR variance term is not positive")
    z_score = (sharpe - float(benchmark_sharpe)) * math.sqrt(observations - 1.0) / math.sqrt(variance_term)
    if not math.isfinite(z_score):
        return _unavailable("PSR score is not finite")
    return _available(
        float(stats.norm.cdf(z_score)),
        sharpe_per_period=sharpe,
        benchmark_sharpe=float(benchmark_sharpe),
        skewness=skew,
        kurtosis_non_excess=kurtosis,
        observations=int(observations),
    )


def expected_maximum_sharpe(trial_sharpes: Sequence[Any], trial_count: int) -> Dict[str, Any]:
    """Euler-Mascheroni hurdle for the maximum Sharpe. The null mean is zero."""

    count = int(trial_count)
    if count < 2:
        return _unavailable("trial count is below 2")
    clean = _finite(trial_sharpes)
    if len(clean) < 2:
        return _unavailable("trial Sharpe variance cannot be computed")
    variance = float(clean.var(ddof=1))
    if not math.isfinite(variance) or variance < 0.0:
        return _unavailable("trial Sharpe variance cannot be computed")
    z_one = float(stats.norm.ppf(1.0 - 1.0 / count))
    z_two = float(stats.norm.ppf(1.0 - 1.0 / (count * math.e)))
    hurdle = math.sqrt(variance) * ((1.0 - EULER_MASCHERONI) * z_one + EULER_MASCHERONI * z_two)
    return _available(hurdle, trial_count=count, trial_sharpe_variance=variance, null_mean=0.0)


def deflated_sharpe_ratio(returns: Sequence[Any], trial_sharpes: Sequence[Any], trial_count: int) -> Dict[str, Any]:
    """PSR at the expected-maximum Sharpe of the recorded trials."""

    hurdle = expected_maximum_sharpe(trial_sharpes, trial_count)
    if not hurdle["available"]:
        return _unavailable(str(hurdle["reason"]))
    psr = probabilistic_sharpe_ratio(returns, benchmark_sharpe=float(hurdle["value"]))
    if not psr["available"]:
        return _unavailable(str(psr["reason"]))
    return _available(
        float(psr["value"]),
        benchmark_sharpe=float(hurdle["value"]),
        trial_count=int(trial_count),
        trial_sharpe_variance=hurdle["trial_sharpe_variance"],
        null_mean=0.0,
    )


def underwater_statistics(equity: Sequence[Any]) -> Dict[str, Any]:
    """Time below the running peak, and the longest such consecutive span."""

    clean = np.asarray(list(equity) if not isinstance(equity, np.ndarray) else equity, dtype=float).reshape(-1)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return {
            "time_under_water_periods": _unavailable("equity series is empty"),
            "drawdown_duration_periods": _unavailable("equity series is empty"),
            "path_type": "single_path",
        }
    peak = np.maximum.accumulate(clean)
    underwater = clean < peak
    total = int(underwater.sum())
    longest = current = 0
    for flag in underwater:
        if bool(flag):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "time_under_water_periods": _available(total),
        "drawdown_duration_periods": _available(longest),
        "path_type": "single_path",
    }


def hhi_concentration(values: Sequence[Any]) -> Dict[str, Any]:
    """Herfindahl-Hirschman index of the positive mass in ``values``."""

    clean = _finite(values)
    positive = clean[clean > 0.0]
    if len(positive) == 0:
        return _unavailable("no positive mass")
    weights = positive / float(positive.sum())
    return _available(float(np.sum(weights ** 2)), observations=int(len(positive)))


def holding_period_distribution(holding_bars: Sequence[Any]) -> Dict[str, Any]:
    clean = _finite(holding_bars)
    clean = clean[clean >= 0.0]
    if len(clean) == 0:
        return {"count": 0, "min": None, "p50": None, "mean": None, "max": None, "histogram": {}, "path_type": "single_path"}
    histogram: Dict[str, int] = {}
    for value in clean.astype(int):
        key = str(int(value))
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "count": int(len(clean)),
        "min": float(np.min(clean)),
        "p50": float(np.quantile(clean, 0.5)),
        "mean": float(np.mean(clean)),
        "max": float(np.max(clean)),
        "histogram": histogram,
        "path_type": "single_path",
    }


def independent_bet_proxy(frame: pd.DataFrame) -> Dict[str, Any]:
    """Effective sample size of trade uniqueness, from ``sample_weights.average_uniqueness``."""

    if frame is None or len(frame) == 0:
        return _unavailable("no trades")
    if {"label_span_start_idx", "label_span_end_idx"} <= set(frame.columns):
        uniqueness = average_uniqueness(frame["label_span_start_idx"], frame["label_span_end_idx"])
        method = "label_span_uniqueness"
    elif "average_uniqueness" in frame.columns:
        uniqueness = _finite(frame["average_uniqueness"])
        method = "average_uniqueness_column"
    else:
        return _unavailable("uniqueness spans are not on the trades")
    if len(uniqueness) == 0:
        return _unavailable("uniqueness spans are not on the trades")
    return _available(
        effective_sample_size(uniqueness),
        average_uniqueness=float(np.mean(uniqueness)),
        method=method,
    )


def pnl_per_turnover(net_pnl: float, turnover: float) -> Dict[str, Any]:
    if not math.isfinite(float(net_pnl)) or not math.isfinite(float(turnover)):
        return _unavailable("net pnl or turnover is not finite")
    if abs(float(turnover)) <= 1e-12:
        return _unavailable("turnover is zero")
    return _available(float(net_pnl) / float(turnover))


def equity_from_period_pnl(period_pnl: Sequence[Any], initial_capital: float) -> np.ndarray:
    pnl = np.asarray(list(period_pnl) if not isinstance(period_pnl, np.ndarray) else period_pnl, dtype=float).reshape(-1)
    pnl = np.where(np.isfinite(pnl), pnl, 0.0)
    if len(pnl) == 0:
        return np.asarray([float(initial_capital)], dtype=float)
    return float(initial_capital) + np.cumsum(pnl)


def probability_of_backtest_overfitting(returns: Sequence[Sequence[Any]], *, slices: Optional[int] = None) -> Dict[str, Any]:
    """CSCV probability that the in-sample winner has a negative out-of-sample mean.

    ``returns`` has shape ``(n_periods, n_trials)``. Full CPCV is not computed here.
    Complementary partitions are not double-counted: slice 0 stays in-sample.
    """

    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        return _unavailable("trial-by-period return matrix must be two-dimensional", path_type="multi_path")
    n_periods, n_trials = matrix.shape
    if n_trials < 2 or n_periods < 8:
        return _unavailable("trial-by-period return matrix is too small for CSCV", path_type="multi_path")
    slice_count = 16 if slices is None else int(slices)
    slice_count = min(slice_count, n_periods)
    if slice_count % 2:
        slice_count -= 1
    if slice_count < 4:
        return _unavailable("trial-by-period return matrix is too small for CSCV", path_type="multi_path")
    usable = (n_periods // slice_count) * slice_count
    groups = np.array_split(matrix[:usable], slice_count)
    half = slice_count // 2
    decisive = 0
    overfit = 0
    for rest in combinations(range(1, slice_count), half - 1):
        in_sample = (0, *rest)
        out_sample = tuple(index for index in range(slice_count) if index not in in_sample)
        in_returns = np.concatenate([groups[index] for index in in_sample], axis=0)
        out_returns = np.concatenate([groups[index] for index in out_sample], axis=0)
        in_mean = np.nanmean(in_returns, axis=0)
        if not np.isfinite(in_mean).all():
            continue
        winner = int(np.argmax(in_mean))
        if int(np.sum(np.isclose(in_mean, in_mean[winner]))) != 1:
            continue
        out_mean = float(np.nanmean(out_returns[:, winner]))
        if not math.isfinite(out_mean):
            continue
        decisive += 1
        if out_mean < 0.0:
            overfit += 1
    if decisive == 0:
        return _unavailable("no CSCV split had a unique in-sample winner", path_type="multi_path")
    return _available(
        overfit / decisive,
        path_type="multi_path",
        method="cscv",
        splits=decisive,
        slices=slice_count,
        trials=int(n_trials),
        periods=int(usable),
    )


def classify_evidence_grade(
    *,
    trade_count: float,
    dsr_available: bool,
    final_test_uses: int,
    smoke: bool,
    min_trades: int,
) -> str:
    """``alpha_claim_grade`` only when every explicit rule holds.

    The trade rule is ``trades >= max(min_trades, 1)``. DSR must be available, the
    final test must have been used once, and the run must not be smoke. Anything
    else is ``debugging_only``.
    """

    required_trades = max(int(min_trades), 1)
    if (not smoke) and float(trade_count) >= required_trades and bool(dsr_available) and int(final_test_uses) == 1:
        return "alpha_claim_grade"
    return "debugging_only"


def summarize_single_path(
    period_pnl: Sequence[Any],
    *,
    initial_capital: float,
    net_pnl: float,
    turnover: float,
    holding_bars: Sequence[Any],
    trades: Optional[pd.DataFrame] = None,
    granularity: str = "5m",
    negative_pnl: Optional[Sequence[Any]] = None,
    positive_pnl: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Single-path statistics. DSR is filled later, once every trial Sharpe exists."""

    pnl = np.asarray(list(period_pnl) if not isinstance(period_pnl, np.ndarray) else period_pnl, dtype=float).reshape(-1)
    pnl = np.where(np.isfinite(pnl), pnl, 0.0)
    capital = float(initial_capital) if initial_capital else 1.0
    period_returns = pnl / capital
    equity = equity_from_period_pnl(pnl, capital)
    positive = pnl[pnl > 0.0] if positive_pnl is None else positive_pnl
    negative = np.abs(pnl[pnl < 0.0]) if negative_pnl is None else negative_pnl
    sharpe = sharpe_ratio(period_returns, granularity=granularity)
    underwater = underwater_statistics(equity)
    return {
        "path_type": "single_path",
        "annualisation": ANNUALISATION_NOTE,
        "periods_per_year": sharpe["periods_per_year"],
        "sharpe_per_period": sharpe["per_period"],
        "sharpe_annualised": sharpe["annualised"],
        "psr": probabilistic_sharpe_ratio(period_returns, benchmark_sharpe=0.0),
        "dsr": _unavailable("trial count was not supplied"),
        "time_under_water_periods": underwater["time_under_water_periods"],
        "drawdown_duration_periods": underwater["drawdown_duration_periods"],
        "hhi_positive_pnl": hhi_concentration(positive),
        "hhi_negative_pnl": hhi_concentration(negative),
        "hhi_time": hhi_concentration(holding_bars),
        "holding_period": holding_period_distribution(holding_bars),
        "independent_bets": independent_bet_proxy(trades if trades is not None else pd.DataFrame()),
        "pnl_per_turnover": pnl_per_turnover(net_pnl, turnover),
        "period_returns": [float(value) for value in period_returns],
    }


def period_pnl_by_timestamp(candidates: pd.DataFrame, trades: pd.DataFrame) -> np.ndarray:
    if candidates is None or candidates.empty or "timestamp" not in candidates.columns:
        return np.asarray([], dtype=float)
    clock = pd.to_datetime(candidates["timestamp"], utc=True).drop_duplicates().sort_values()
    pnl = pd.Series(0.0, index=clock)
    if trades is not None and not trades.empty and "timestamp" in trades.columns and "net_pnl" in trades.columns:
        grouped = trades.copy()
        grouped["timestamp"] = pd.to_datetime(grouped["timestamp"], utc=True)
        summed = grouped.groupby("timestamp")["net_pnl"].sum()
        pnl = pnl.add(summed, fill_value=0.0).reindex(clock).fillna(0.0)
    return pnl.to_numpy(dtype=float)


def attach_cross_trial_statistics(trials: Sequence[Dict[str, Any]], *, trial_count: int) -> Dict[str, Any]:
    """Fill DSR from the recorded trial Sharpes and return the CSCV PBO object."""

    sharpes = []
    matrix_rows = []
    for trial in trials:
        if trial.get("status") != "completed":
            continue
        validation = ((trial.get("metrics") or {}).get("validation") or {}).get("statistics") or {}
        sharpe = (validation.get("sharpe_per_period") or {}).get("value")
        if isinstance(sharpe, (int, float)) and math.isfinite(float(sharpe)):
            sharpes.append(float(sharpe))
        returns = validation.get("period_returns") or []
        if returns:
            matrix_rows.append([float(value) for value in returns])
    for trial in trials:
        metrics = trial.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for split in ("validation", "test"):
            block = metrics.get(split) or {}
            stats_block = block.get("statistics") if isinstance(block, dict) else None
            if not isinstance(stats_block, dict) or "period_returns" not in stats_block:
                continue
            stats_block["dsr"] = deflated_sharpe_ratio(stats_block["period_returns"], sharpes, trial_count)
    lengths = {len(row) for row in matrix_rows}
    if len(matrix_rows) >= 2 and len(lengths) == 1:
        pbo = probability_of_backtest_overfitting(np.asarray(matrix_rows, dtype=float).T)
    elif len(matrix_rows) < 2:
        pbo = _unavailable("trial-by-period return matrix is too small for CSCV", path_type="multi_path")
    else:
        pbo = _unavailable("trial period-return series have different lengths", path_type="multi_path")
    return {"pbo": pbo, "path_type": "single_path"}
