import pytest

from core.schema import (
    SchemaError, extract_json, normalize_agent_output, normalize_confidence,
    normalize_forecasts, normalize_orders,
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


def test_orders_accept_the_whiteboard_positional_form():
    (order,) = normalize_orders([["Buy", 10, "AAPL"]], universe=["AAPL"])
    assert order["side"] == "BUY" and order["qty"] == 10 and order["ticker"] == "AAPL"


def test_orders_accept_the_dict_form_with_aliases():
    (order,) = normalize_orders([{"action": "sell", "shares": "5", "stock": "qqq"}],
                                universe=["QQQ"])
    assert order["side"] == "SELL" and order["qty"] == 5.0


def test_hold_is_not_an_order():
    assert normalize_orders([{"side": "HOLD", "ticker": "QQQ", "qty": 5}], universe=["QQQ"]) == []


def test_orders_for_unknown_tickers_are_dropped():
    assert normalize_orders([{"side": "BUY", "ticker": "MOON", "qty": 5}], universe=["QQQ"]) == []


def test_non_positive_quantities_are_dropped():
    assert normalize_orders([{"side": "BUY", "ticker": "QQQ", "qty": 0}], universe=["QQQ"]) == []
    assert normalize_orders([{"side": "BUY", "ticker": "QQQ", "qty": -5}],
                            universe=["QQQ"])[0]["qty"] == 5.0     # sign lives in `side`


def test_duplicate_orders_on_the_same_side_are_merged():
    orders = normalize_orders([{"side": "BUY", "ticker": "QQQ", "qty": 5},
                               {"side": "BUY", "ticker": "QQQ", "qty": 3}], universe=["QQQ"])
    assert len(orders) == 1 and orders[0]["qty"] == 8


def test_a_limit_order_without_a_price_becomes_a_market_order():
    (order,) = normalize_orders([{"side": "BUY", "ticker": "QQQ", "qty": 5,
                                  "order_type": "LIMIT"}], universe=["QQQ"])
    assert order["order_type"] == "MARKET" and order["limit_price"] is None


def test_agent_output_carries_orders_from_either_key():
    out = normalize_agent_output({"actions": [{"side": "BUY", "ticker": "SPY", "qty": 2}]},
                                 agent_name="a", persona="p", timestamp="t", universe=["SPY"])
    assert out["orders"][0]["ticker"] == "SPY"
