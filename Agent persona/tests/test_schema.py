import pytest

from core.schema import (
    SchemaError, extract_json, normalize_agent_output, normalize_confidence,
    normalize_forecasts,
)


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(SchemaError):
        extract_json("no json here")


def test_confidence_accepts_every_scale_models_emit():
    assert normalize_confidence(0.8) == 0.8            # as asked
    assert normalize_confidence(8) == 0.8              # ten-point scale
    assert normalize_confidence(80) == 0.8             # percent
    assert normalize_confidence("65%") == 0.65
    assert normalize_confidence(9999) == 1.0           # clamped
    assert normalize_confidence(-1) == 0.0


def test_hallucinated_tickers_are_dropped():
    out = normalize_forecasts(
        [{"ticker": "MOON", "direction": "UP", "expected_return_bps": 50, "confidence": 0.9}],
        universe=["SPY"],
    )
    assert out == []


def test_direction_wins_over_contradictory_magnitude():
    (f,) = normalize_forecasts(
        [{"ticker": "spy", "direction": "bearish", "expected_return_bps": "45", "confidence": 0.5}],
        universe=["SPY"],
    )
    assert f["ticker"] == "SPY"
    assert f["direction"] == "DOWN"
    assert f["expected_return_bps"] < 0


def test_missing_direction_is_inferred_from_magnitude():
    (f,) = normalize_forecasts([{"ticker": "SPY", "expected_return_bps": -3}], universe=["SPY"])
    assert f["direction"] == "FLAT"  # inside the flat band


def test_duplicate_tickers_keep_the_most_confident():
    out = normalize_forecasts(
        [
            {"ticker": "SPY", "direction": "UP", "expected_return_bps": 20, "confidence": 0.2},
            {"ticker": "SPY", "direction": "DOWN", "expected_return_bps": -30, "confidence": 0.8},
        ],
        universe=["SPY"],
    )
    assert len(out) == 1 and out[0]["direction"] == "DOWN"


def test_agent_output_fills_required_keys():
    out = normalize_agent_output(
        {"forecasts": [{"ticker": "SPY", "direction": "UP", "expected_return_bps": 30}]},
        agent_name="a", persona="p", timestamp="t", universe=["SPY"],
    )
    for key in ("agent_name", "persona", "timestamp", "market_view", "signals",
                "forecasts", "trade_ideas", "checks"):
        assert key in out
    assert out["market_view"]["risk_regime"] == "NEUTRAL"
