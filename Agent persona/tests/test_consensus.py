from orchestration.consensus import build_consensus


def _agent(name, forecasts, regime="NEUTRAL", conf=0.5):
    return {
        "agent_name": name, "persona": name, "timestamp": "t", "decision": "",
        "market_view": {"risk_regime": regime, "confidence": conf},
        "signals": [], "forecasts": forecasts, "trade_ideas": [], "checks": {},
    }


def f(ticker, direction, bps, conf):
    return {"ticker": ticker, "direction": direction, "expected_return_bps": bps,
            "horizon": "1d", "confidence": conf, "rationale": "", "news_refs": []}


def test_unanimous_direction_gives_full_agreement():
    c = build_consensus([
        _agent("a", [f("SPY", "DOWN", -40, 0.8)]),
        _agent("b", [f("SPY", "DOWN", -60, 0.6)]),
    ])
    (spy,) = c["forecasts"]
    assert spy["direction"] == "DOWN"
    assert spy["agreement"] == 1.0
    assert -60 < spy["expected_return_bps"] < -40      # confidence-weighted mean
    assert spy["dispersion_bps"] > 0


def test_unanimous_flat_is_agreement_not_a_coin_flip():
    c = build_consensus([_agent("a", [f("SPY", "FLAT", 0, 0.5)]),
                         _agent("b", [f("SPY", "FLAT", 2, 0.5)])])
    (spy,) = c["forecasts"]
    assert spy["direction"] == "FLAT"
    assert spy["agreement"] == 1.0
    assert spy["confidence"] > 0


def test_split_panel_is_discounted():
    c = build_consensus([
        _agent("a", [f("SPY", "UP", 50, 0.7)]),
        _agent("b", [f("SPY", "DOWN", -50, 0.7)]),
    ])
    (spy,) = c["forecasts"]
    assert spy["direction"] == "FLAT"          # no directional call from a tied panel
    assert spy["agreement"] < 1.0
    assert spy["confidence"] < 0.7


def test_confidence_weighting_moves_the_consensus():
    c = build_consensus([
        _agent("loud", [f("QQQ", "UP", 80, 0.9)]),
        _agent("quiet", [f("QQQ", "DOWN", -20, 0.1)]),
    ])
    (qqq,) = c["forecasts"]
    assert qqq["direction"] == "UP"
    assert qqq["net_vote"] > 0.5


def test_regime_vote_uses_stated_confidence():
    c = build_consensus([
        _agent("a", [f("SPY", "DOWN", -30, 0.5)], regime="RISK_OFF", conf=0.9),
        _agent("b", [f("SPY", "UP", 30, 0.5)], regime="RISK_ON", conf=0.2),
    ])
    assert c["risk_regime"] == "RISK_OFF"


def test_zero_confidence_panel_does_not_divide_by_zero():
    c = build_consensus([_agent("a", [f("SPY", "UP", 30, 0.0)]),
                         _agent("b", [f("SPY", "UP", 30, 0.0)])])
    assert c["forecasts"][0]["direction"] == "UP"
