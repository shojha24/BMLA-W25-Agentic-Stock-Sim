import pytest

from data.market_data import PriceStore
from eval.benchmarks import DEFAULT_BENCHMARKS, build_benchmarks

UNIVERSE = ["AAA", "BBB"]


@pytest.fixture
def prices(tmp_path):
    store = PriceStore(cache_dir=tmp_path)
    (tmp_path / "AAA.csv").write_text("date,close\n2016-01-04,100.0\n2016-01-05,105.0\n")
    (tmp_path / "BBB.csv").write_text("date,close\n2016-01-04,100.0\n2016-01-05,95.0\n")
    return store


@pytest.fixture
def digest():
    return {"timestamp": "2016-01-05T21:00:00Z", "news_digest": [
        {"news_id": "n1", "headline": "h", "summary": "", "tickers_mentioned": ["AAA"],
         "macro_tags": ["GENERAL"], "sentiment": "BEARISH", "confidence": 0.8}]}


@pytest.fixture
def state():
    return {"cash_usd": 10000.0, "positions": {}, "prices": {"AAA": 105.0, "BBB": 95.0}}


def run(name, prices, digest, state, day="2016-01-05"):
    (bench,) = build_benchmarks([name])
    return bench.forecast(digest=digest, state=state, universe=UNIVERSE, day=day,
                          prices=prices, horizon="1d")


def test_every_benchmark_emits_the_forecast_contract(prices, digest, state):
    for name in DEFAULT_BENCHMARKS:
        for f in run(name, prices, digest, state):
            assert set(f) == {"ticker", "direction", "expected_return_bps", "horizon",
                              "confidence", "rationale", "news_refs"}
            assert f["direction"] in ("UP", "DOWN", "FLAT")
            assert 0.0 <= f["confidence"] <= 1.0


def test_persistence_follows_the_prior_session(prices, digest, state):
    calls = {f["ticker"]: f["direction"] for f in run("persistence", prices, digest, state)}
    assert calls == {"AAA": "UP", "BBB": "DOWN"}      # AAA rose, BBB fell


def test_reversal_is_the_mirror_of_persistence(prices, digest, state):
    calls = {f["ticker"]: f["direction"] for f in run("reversal", prices, digest, state)}
    assert calls == {"AAA": "DOWN", "BBB": "UP"}


def test_persistence_is_flat_without_a_prior_close(prices, digest, state):
    calls = {f["ticker"]: f["direction"] for f in run("persistence", prices, digest, state,
                                                      day="2016-01-04")}
    assert set(calls.values()) == {"FLAT"}


def test_always_long_is_unconditionally_up(prices, digest, state):
    assert all(f["direction"] == "UP" for f in run("always_long", prices, digest, state))


def test_random_is_seeded_and_reproducible(prices, digest, state):
    first = [f["direction"] for f in run("random", prices, digest, state)]
    second = [f["direction"] for f in run("random", prices, digest, state)]
    assert first == second


def test_sentiment_rule_follows_the_digest(prices, digest, state):
    calls = {f["ticker"]: f["direction"] for f in run("sentiment_rule", prices, digest, state)}
    assert calls  # bearish digest, positive-beta names -> not long
    assert all(d in ("DOWN", "FLAT") for d in calls.values())


def test_unknown_benchmark_is_rejected():
    with pytest.raises(ValueError):
        build_benchmarks(["magic_eight_ball"])
