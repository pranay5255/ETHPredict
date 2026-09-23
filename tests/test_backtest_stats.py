from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import stats

from src.evaluation.backtest_stats import (
    FIVE_MINUTE_BARS_PER_YEAR,
    classify_evidence_grade,
    deflated_sharpe_ratio,
    hhi_concentration,
    probability_of_backtest_overfitting,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    underwater_statistics,
)
from src.experiments.staged_trial import run_staged_trial
from tests.test_staged_trial import _v2_config, _write_5m_ohlcv


STATISTIC_FIELDS = (
    "path_type",
    "sharpe_per_period",
    "sharpe_annualised",
    "annualisation",
    "psr",
    "dsr",
    "time_under_water_periods",
    "drawdown_duration_periods",
    "hhi_positive_pnl",
    "hhi_negative_pnl",
    "hhi_time",
    "holding_period",
    "independent_bets",
    "pnl_per_turnover",
)


def test_sharpe_uses_the_five_minute_24_7_factor():
    returns = np.array([0.01, -0.01, 0.02, 0.0])
    result = sharpe_ratio(returns)

    expected = float(returns.mean() / returns.std(ddof=1))
    assert result["per_period"]["value"] == pytest.approx(expected)
    assert result["annualised"]["value"] == pytest.approx(expected * np.sqrt(FIVE_MINUTE_BARS_PER_YEAR))
    assert result["periods_per_year"] == FIVE_MINUTE_BARS_PER_YEAR
    assert "5-minute" in result["annualisation"]
    assert "24/7" in result["annualisation"]


def test_psr_matches_the_bailey_lopez_de_prado_cdf():
    returns = np.random.default_rng(1).normal(0.002, 0.01, size=400)
    result = probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)

    sharpe = float(returns.mean() / returns.std(ddof=1))
    skew = float(stats.skew(returns, bias=False))
    kurtosis = float(stats.kurtosis(returns, fisher=False, bias=False))
    variance_term = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    expected = float(stats.norm.cdf((sharpe - 0.0) * np.sqrt(len(returns) - 1) / np.sqrt(variance_term)))

    assert result["value"] == pytest.approx(expected, abs=1e-8)
    assert result["kurtosis_non_excess"] == pytest.approx(kurtosis)


def test_dsr_moves_with_trial_count_and_is_unavailable_below_two_trials():
    returns = np.random.default_rng(2).normal(0.001, 0.01, size=300)
    trial_sharpes = [0.2, 0.0, -0.1, 0.05]

    small = deflated_sharpe_ratio(returns, trial_sharpes, 2)
    large = deflated_sharpe_ratio(returns, trial_sharpes, 40)
    missing = deflated_sharpe_ratio(returns, trial_sharpes, 1)

    assert small["available"] is True
    assert large["available"] is True
    assert large["value"] < small["value"]
    assert missing["value"] == "unavailable"
    assert "below 2" in missing["reason"]


def test_dsr_reports_when_trial_sharpe_variance_cannot_be_computed():
    returns = np.array([0.01, -0.01, 0.02, 0.0, 0.01])
    result = deflated_sharpe_ratio(returns, [0.4], 4)

    assert result["value"] == "unavailable"
    assert "variance" in result["reason"]


def test_time_under_water_counts_the_span_below_the_peak():
    result = underwater_statistics([10.0, 12.0, 11.0, 9.0, 10.0, 13.0])

    assert result["time_under_water_periods"]["value"] == 3
    assert result["drawdown_duration_periods"]["value"] == 3


def test_positive_pnl_hhi_is_one_when_one_observation_holds_all_positive_pnl():
    result = hhi_concentration([5.0, 0.0, -1.0])

    assert result["value"] == pytest.approx(1.0)


def test_pbo_is_high_for_an_overfit_matrix_and_placeholder_when_too_small():
    column_a = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    column_b = -0.5 * column_a
    overfit = probability_of_backtest_overfitting(np.column_stack([column_a, column_b]), slices=4)
    placeholder = probability_of_backtest_overfitting(np.ones((3, 2)))

    assert overfit["path_type"] == "multi_path"
    assert overfit["value"] > 0.5
    assert placeholder["value"] == "unavailable"
    assert "too small" in placeholder["reason"]
    assert placeholder["path_type"] == "multi_path"


def test_evidence_grade_requires_every_alpha_claim_rule():
    claim = dict(trade_count=10, dsr_available=True, final_test_uses=1, smoke=False, min_trades=1)
    assert classify_evidence_grade(**claim) == "alpha_claim_grade"
    assert classify_evidence_grade(**{**claim, "smoke": True}) == "debugging_only"
    assert classify_evidence_grade(**{**claim, "dsr_available": False}) == "debugging_only"
    assert classify_evidence_grade(**{**claim, "final_test_uses": 2}) == "debugging_only"
    assert classify_evidence_grade(**{**claim, "trade_count": 0}) == "debugging_only"


def test_v2_smoke_report_includes_backtest_statistics(tmp_path):
    data_dir = tmp_path / "data"
    _write_5m_ohlcv(data_dir / "raw" / "ETHUSDT-5m-lighter-20260328-20260628.csv")
    config = _v2_config(data_dir, tmp_path / "runs")
    config["benchmark"] = {"enabled": False}
    config_path = tmp_path / "v2.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_staged_trial(config_path, smoke=True, device="cpu", allow_cpu=True)

    assert result["path_type"] == "single_path"
    assert result["evidence_grade"] == "debugging_only"
    assert result["pbo"]["path_type"] == "multi_path"
    assert result["pbo"]["value"] == "unavailable"
    assert result["pbo"]["reason"]
    assert "5-minute" in result["annualisation"]
    for split in ("validation", "test"):
        statistics = result["best_trial"]["metrics"][split]["statistics"]
        assert statistics["path_type"] == "single_path"
        for field in STATISTIC_FIELDS:
            assert field in statistics
        assert statistics["path_type"] == "single_path"
    markdown = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
    assert "debugging_only" in markdown
    assert "5-minute" in markdown
