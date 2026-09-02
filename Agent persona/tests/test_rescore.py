"""Scoring a run after the fact - the only way a live forecast can be judged."""
import json

import pytest

from data.market_data import PriceStore
from eval.rescore import load_cycles, rescore_run


def cycle(day, timestamp, direction="UP", bps=50.0, ticker="AAA"):
    forecast = {"ticker": ticker, "direction": direction, "expected_return_bps": bps,
                "horizon": "15m", "confidence": 0.7, "rationale": "", "news_refs": []}
    return {
        "run_id": "run_live_1", "day": day, "timestamp": timestamp,
        "prices": {ticker: 100.0, "SPY": 400.0},
        "consensus": {"risk_regime": "RISK_ON", "horizon": "15m", "forecasts": [forecast]},
        "consensus_round1": {"forecasts": [forecast]},
        "agent_outputs": [{"agent_name": "macro", "forecasts": [forecast]}],
        "realized_bps": {}, "fills": [],
    }


@pytest.fixture
def intraday_prices(tmp_path):
    """A cache of 15-minute bars: AAA rising, SPY rising."""
    store = PriceStore(cache_dir=tmp_path)
    directory = tmp_path / "intraday"
    directory.mkdir(parents=True, exist_ok=True)
    stamps = [f"2026-09-02T15:{m:02d}:00Z" for m in (0, 15, 30, 45)]
    for ticker, series in (("AAA", [100.0, 101.0, 102.0, 103.0]),
                           ("SPY", [400.0, 401.0, 402.0, 403.0])):
        rows = "\n".join(f"{t},{p}" for t, p in zip(stamps, series))
        (directory / f"{ticker}_15m.csv").write_text("timestamp,close\n" + rows + "\n")
    return store


@pytest.fixture
def live_log(tmp_path):
    path = tmp_path / "run_live_1.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in [
        cycle("2026-09-02", "2026-09-02T15:00:00Z"),
        cycle("2026-09-02", "2026-09-02T15:15:00Z", direction="DOWN", bps=-40.0),
        cycle("2026-09-02", "2026-09-02T15:45:00Z"),      # no later bar: unscorable
    ]) + "\n")
    return path


def test_a_live_run_can_be_scored_after_the_fact(live_log, intraday_prices):
    report = rescore_run(live_log, intraday_prices, horizon="15m")
    assert report["cycles_in_log"] == 3
    assert report["cycles_scored"] == 2
    assert report["cycles_not_yet_scorable"] == 1        # last cycle has no forward bar
    assert report["bar_interval"] == "15m"


def test_scoring_judges_direction_against_the_next_bar(live_log, intraday_prices):
    report = rescore_run(live_log, intraday_prices, horizon="15m")
    consensus = report["forecast_metrics"]["agents_consensus"]
    assert consensus["n_directional"] == 2
    assert consensus["hit_rate"] == 0.5                  # UP right, DOWN wrong on a rising tape


def test_every_model_in_the_log_is_scored(live_log, intraday_prices):
    metrics = rescore_run(live_log, intraday_prices, horizon="15m")["forecast_metrics"]
    assert {"agents_consensus", "agents_round1", "agent:macro"} <= set(metrics)


def test_sim_vs_actual_compares_the_regime_call_to_the_index(live_log, intraday_prices):
    sva = rescore_run(live_log, intraday_prices, horizon="15m")["sim_vs_actual"]
    assert sva["regime_calls"] == 2
    assert sva["regime_accuracy"] == 1.0                 # RISK_ON, and SPY rose both times
    assert sva["directional_accuracy"] == 0.5
    assert sva["magnitude_bias_bps"] != 0


def test_the_horizon_recorded_in_the_log_is_used_by_default(live_log, intraday_prices):
    assert rescore_run(live_log, intraday_prices)["horizon"] == "15m"


def test_a_longer_horizon_needs_more_bars_ahead(live_log, intraday_prices):
    """Bars at :00 :15 :30 :45 - a 30m horizon needs two bars ahead of the cycle."""
    report = rescore_run(live_log, intraday_prices, horizon="30m")
    assert report["cycles_scored"] == 2                  # cycles at :00 and :15
    assert report["cycles_not_yet_scorable"] == 1        # the :45 cycle has nowhere to look


def test_missing_run_log_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        rescore_run(tmp_path / "nope.jsonl")


def test_load_cycles_skips_blank_lines(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(cycle("2026-09-02", "2026-09-02T15:00:00Z")) + "\n\n")
    assert len(load_cycles(path)) == 1
