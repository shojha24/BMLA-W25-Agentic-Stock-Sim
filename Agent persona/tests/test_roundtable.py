import pytest

from agents.base import BaseAgent
from agents.baseline import SentimentBaselineAgent
from agents.persona import LLMPersonaAgent
from agents.personas import PERSONAS
from llm.mock_client import MockChatClient
from orchestration.roundtable import Roundtable, revision_deltas


class _NoRag:
    def retrieve(self, *a, **k):
        return [], ""

    def status(self):
        return {"mode": "disabled"}


def build_panel(model="mock"):
    client = MockChatClient()
    return [
        LLMPersonaAgent(spec=PERSONAS[key], client=client, model=model, rag_tool=_NoRag())
        for key in ("macro_econ", "quant_momentum", "contrarian_value")
    ]


class ExplodingAgent(BaseAgent):
    name = "exploding_agent"
    persona = "always fails"

    def run(self, digest, state, peer_context=None, prior_output=None):
        raise RuntimeError("rate limited")


def test_two_rounds_produce_a_consensus(digest, state):
    result = Roundtable(build_panel(), rounds=2).run(digest, state)
    assert len(result["rounds"]) == 2
    assert result["errors"] == []
    assert result["consensus"]["forecasts"]
    assert set(result["consensus"]["agents"]) == {
        "macro_econ_llm_v2", "quant_momentum_llm_v1", "contrarian_value_llm_v1"
    }


def test_round1_consensus_is_kept_for_ablation(digest, state):
    result = Roundtable(build_panel(), rounds=2).run(digest, state)
    assert result["consensus_round1"]["forecasts"]
    assert result["revision"]["compared_forecasts"] > 0


def test_personas_disagree(digest, state):
    """A panel whose members always agree adds nothing over one agent."""
    result = Roundtable(build_panel(), rounds=1).run(digest, state)
    directions = {
        out["agent_name"]: {f["ticker"]: f["direction"] for f in out["forecasts"]}
        for out in result["rounds"][0]["outputs"]
    }
    per_ticker = {}
    for calls in directions.values():
        for ticker, d in calls.items():
            per_ticker.setdefault(ticker, set()).add(d)
    assert any(len(v) > 1 for v in per_ticker.values())


def test_one_failing_agent_does_not_kill_the_panel(digest, state):
    agents = build_panel() + [ExplodingAgent()]
    result = Roundtable(agents, rounds=2).run(digest, state)
    assert result["errors"]
    assert all(e["agent"] == "exploding_agent" for e in result["errors"])
    assert result["consensus"]["forecasts"]


def test_all_agents_failing_raises(digest, state):
    with pytest.raises(RuntimeError):
        Roundtable([ExplodingAgent()], rounds=1).run(digest, state)


def test_agents_see_peers_only_in_round_two(digest, state):
    client = MockChatClient()
    agents = [
        LLMPersonaAgent(spec=PERSONAS[k], client=client, model="mock", rag_tool=_NoRag())
        for k in ("macro_econ", "contrarian_value")
    ]
    Roundtable(agents, rounds=2, parallel=False).run(digest, state)
    payloads = [c["messages"][-1]["content"] for c in client.calls]
    assert all("peers" not in p for p in payloads[:2])
    assert all("peers" in p for p in payloads[2:])
    # and never its own forecast presented as a peer's
    assert "macro_econ_llm_v2" not in payloads[2]


def test_revision_deltas_detect_flips():
    def out(name, direction, bps, conf):
        return {"agent_name": name, "market_view": {}, "forecasts": [
            {"ticker": "SPY", "direction": direction, "expected_return_bps": bps,
             "horizon": "1d", "confidence": conf, "rationale": "", "news_refs": []}]}

    d = revision_deltas([out("a", "UP", 40, 0.6)], [out("a", "DOWN", -20, 0.5)])
    assert d["direction_flips"] == 1
    assert d["flip_rate"] == 1.0
    assert d["mean_abs_delta_bps"] == 60.0
    assert d["mean_delta_confidence"] == pytest.approx(-0.1)


def test_baseline_agent_runs_without_network(digest, state):
    out = SentimentBaselineAgent().run(digest, state)
    assert out["forecasts"]
    assert {f["ticker"] for f in out["forecasts"]} == set(state["prices"])


def test_baseline_agent_places_no_orders_on_a_mixed_digest(digest, state):
    """One bullish and one bearish item nets to no view, so it should stand aside."""
    assert SentimentBaselineAgent().run(digest, state)["orders"] == []


def test_baseline_agent_emits_valid_order_records(digest, state):
    """Its orders reach the venue directly, so they must satisfy the Order contract."""
    decisive = {**digest, "news_digest": [
        {**digest["news_digest"][0], "sentiment": "BEARISH", "confidence": 0.9}]}
    out = SentimentBaselineAgent().run(decisive, state)
    assert out["orders"]
    for order in out["orders"]:
        assert set(order) == {"ticker", "side", "qty", "order_type", "limit_price",
                              "rationale", "news_refs"}
        assert order["side"] in ("BUY", "SELL")
        assert order["qty"] > 0
        assert order["ticker"] in state["prices"]
