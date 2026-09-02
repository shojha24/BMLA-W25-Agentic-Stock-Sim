import pytest

from sim.portfolio import Portfolio, target_weights


def f(ticker, direction, conf, bps=50.0):
    return {"ticker": ticker, "direction": direction, "confidence": conf,
            "expected_return_bps": bps, "horizon": "1d"}


def test_weights_scale_direction_by_confidence():
    w = target_weights([f("QQQ", "UP", 0.8), f("TLT", "DOWN", 0.4)], max_gross=10.0, max_name=1.0)
    assert w["QQQ"] > 0 and w["TLT"] < 0
    assert abs(w["QQQ"]) > abs(w["TLT"])


def test_low_confidence_and_flat_calls_are_not_traded():
    w = target_weights([f("QQQ", "FLAT", 0.9), f("TLT", "UP", 0.05)])
    assert w == {}


def test_per_name_cap_binds_before_confidence_ordering():
    w = target_weights([f("QQQ", "UP", 0.8), f("TLT", "DOWN", 0.4)], max_gross=10.0, max_name=0.34)
    assert w["QQQ"] == pytest.approx(0.34)
    assert w["TLT"] == pytest.approx(-0.34)


def test_gross_exposure_is_capped():
    w = target_weights([f(t, "UP", 0.9) for t in ("A", "B", "C", "D")], max_gross=1.0, max_name=0.9)
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0)


def test_long_only_drops_shorts():
    w = target_weights([f("QQQ", "DOWN", 0.8)], allow_short=False)
    assert w == {}


def test_rebalance_moves_to_the_target_weight():
    pf = Portfolio(cash_usd=10000.0, cost_bps=0.0)
    pf.rebalance_to({"QQQ": 0.5}, {"QQQ": 100.0})
    assert pf.positions["QQQ"]["qty"] == pytest.approx(50.0)
    assert pf.cash == pytest.approx(5000.0)
    assert pf.equity({"QQQ": 100.0}) == pytest.approx(10000.0)


def test_costs_are_charged_on_traded_notional():
    free = Portfolio(10000.0, cost_bps=0.0)
    paid = Portfolio(10000.0, cost_bps=10.0)      # 10 bps
    for pf in (free, paid):
        pf.rebalance_to({"QQQ": 1.0}, {"QQQ": 100.0})
    assert free.equity({"QQQ": 100.0}) - paid.equity({"QQQ": 100.0}) == pytest.approx(10.0)


def test_price_moves_flow_into_equity():
    pf = Portfolio(10000.0, cost_bps=0.0)
    pf.rebalance_to({"QQQ": 1.0}, {"QQQ": 100.0})
    assert pf.equity({"QQQ": 110.0}) == pytest.approx(11000.0)


def test_short_position_gains_when_price_falls():
    pf = Portfolio(10000.0, cost_bps=0.0)
    pf.rebalance_to({"QQQ": -0.5}, {"QQQ": 100.0})
    assert pf.positions["QQQ"]["qty"] < 0
    assert pf.equity({"QQQ": 90.0}) > 10000.0


def test_state_export_matches_the_agent_contract():
    pf = Portfolio(10000.0)
    pf.rebalance_to({"QQQ": 0.5}, {"QQQ": 100.0})
    state = pf.to_state({"QQQ": 100.0})
    assert set(state) == {"cash_usd", "positions", "prices"}
    assert state["positions"]["QQQ"]["qty"] > 0


def test_dust_trades_are_skipped():
    pf = Portfolio(10000.0, cost_bps=0.0)
    pf.rebalance_to({"QQQ": 0.5}, {"QQQ": 100.0})
    trades = pf.rebalance_to({"QQQ": 0.5000001}, {"QQQ": 100.0})
    assert trades == []
