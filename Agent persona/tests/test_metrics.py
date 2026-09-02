import pytest

from eval.metrics import score_forecasts, summarize_equity, summarize_forecasts


def f(ticker, direction, bps, conf):
    return {"ticker": ticker, "direction": direction, "expected_return_bps": bps,
            "confidence": conf, "horizon": "1d"}


def test_scoring_marks_direction_against_the_realized_move():
    scored = score_forecasts([f("QQQ", "UP", 50, 0.8), f("TLT", "DOWN", -30, 0.6)],
                             {"QQQ": 120.0, "TLT": 40.0})
    assert scored[0]["sign_correct"] is True
    assert scored[1]["sign_correct"] is False
    assert scored[0]["abs_error_bps"] == pytest.approx(70.0)


def test_flat_forecasts_are_not_counted_as_directional():
    scored = score_forecasts([f("QQQ", "FLAT", 0, 0.5)], {"QQQ": 5.0})
    assert scored[0]["sign_correct"] is None
    assert summarize_forecasts(scored)["n_directional"] == 0


def test_tickers_without_a_realized_move_are_dropped():
    assert score_forecasts([f("ZZZ", "UP", 10, 0.5)], {"QQQ": 10.0}) == []


def test_confident_and_wrong_scores_worse_than_unsure_and_wrong():
    confident = summarize_forecasts(score_forecasts([f("QQQ", "UP", 50, 0.95)], {"QQQ": -100.0}))
    unsure = summarize_forecasts(score_forecasts([f("QQQ", "UP", 50, 0.30)], {"QQQ": -100.0}))
    assert confident["brier_score"] > unsure["brier_score"]


def test_equity_summary_reports_drawdown_and_sharpe():
    stats = summarize_equity([100.0, 110.0, 105.0, 120.0])
    assert stats["cumulative_return"] == pytest.approx(0.2)
    assert stats["max_drawdown"] < 0
    assert stats["sharpe"] is not None


def test_short_windows_do_not_report_annualized_returns():
    """Annualizing a two-week sample yields nonsense like +3678%."""
    short = summarize_equity([100.0 * (1.01 ** i) for i in range(10)])
    long = summarize_equity([100.0 * (1.001 ** i) for i in range(120)])
    assert short["annualized_return"] is None
    assert long["annualized_return"] is not None


def test_flat_curve_has_no_sharpe():
    assert summarize_equity([100.0, 100.0, 100.0])["sharpe"] is None
