import json

import pytest

from agents.reflection import DayRecord, ReflectionAgent
from tools.reflection_store import ReflectionStore


def record(**kw):
    base = dict(
        agent_id="macro", persona="Macro Economist", day="2016-06-23", timestamp="T",
        trades=[{"ticker": "GLD", "side": "BUY", "filled_qty": 100, "price": 125.0,
                 "status": "FILLED", "reason": ""}],
        forecasts=[{"ticker": "GLD", "direction": "UP", "expected_return_bps": 80,
                    "confidence": 0.7},
                   {"ticker": "QQQ", "direction": "DOWN", "expected_return_bps": -40,
                    "confidence": 0.6}],
        equity=100000.0, positions={"GLD": {"qty": 100, "avg_price": 125.0}},
    )
    base.update(kw)
    return DayRecord(**base)


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.seen = None

    def chat(self, model, messages, temperature=0.0):
        self.seen = messages
        return json.dumps(self.payload)


class BrokenClient:
    def chat(self, **kwargs):
        raise RuntimeError("model down")


# ---------------- the reflection itself ----------------

def test_heuristic_reflection_scores_trades_against_the_market():
    out = ReflectionAgent(use_llm=False).reflect(record(), {"GLD": 210.0, "QQQ": 55.0}, 101500.0)
    assert "1/1 trades moved my way" in out["lesson"]
    assert out["what_worked"] == ["GLD BUY"]
    assert out["pnl_usd"] == pytest.approx(1500.0)
    assert out["tags"] == ["PROFIT"]


def test_a_losing_trade_is_recorded_as_such():
    out = ReflectionAgent(use_llm=False).reflect(record(), {"GLD": -300.0}, 98000.0)
    assert out["what_failed"] == ["GLD BUY"]
    assert out["tags"] == ["LOSS"]


def test_rejected_orders_and_their_reasons_reach_the_lesson():
    rec = record(trades=[{"ticker": "TLT", "side": "SELL", "filled_qty": 0, "status": "REJECTED",
                          "reason": "insufficient shares (shorting disabled)"}])
    out = ReflectionAgent(use_llm=False).reflect(rec, {"TLT": 40.0}, 100000.0)
    assert "refused" in out["lesson"] and "shorting disabled" in out["lesson"]


def test_a_day_with_no_trades_and_no_positions_produces_nothing():
    rec = record(trades=[], positions={})
    assert ReflectionAgent(use_llm=False).reflect(rec, {"GLD": 10.0}, 100000.0) is None


def test_holding_a_position_is_still_worth_reflecting_on():
    rec = record(trades=[])
    assert ReflectionAgent(use_llm=False).reflect(rec, {"GLD": 10.0}, 100100.0) is not None


def test_llm_reflection_is_used_and_carries_its_persona():
    client = FakeClient({"lesson": "I was right on gold but sized it too small.",
                         "what_worked": ["direction"], "what_failed": ["sizing"],
                         "tags": ["EVENT_RISK"], "tickers": ["GLD"]})
    out = ReflectionAgent(client, "m", use_llm=True).reflect(record(), {"GLD": 210.0}, 101500.0)
    assert out["source"] == "llm"
    assert "sized it too small" in out["lesson"]
    assert out["tags"] == ["EVENT_RISK"]
    assert "Macro Economist" in client.seen[1]["content"]


def test_the_model_is_shown_what_the_market_actually_did():
    client = FakeClient({"lesson": "x"})
    ReflectionAgent(client, "m", use_llm=True).reflect(record(), {"GLD": 210.0}, 101500.0)
    payload = json.loads(client.seen[-1]["content"])
    assert payload["market_moves_bps"]["GLD"] == 210.0
    assert payload["your_trades"][0]["verdict"] == "right"
    assert payload["your_forecasts"][0]["correct"] is True


def test_llm_failure_falls_back_to_the_heuristic_reflection():
    out = ReflectionAgent(BrokenClient(), "m", use_llm=True).reflect(
        record(), {"GLD": 210.0}, 101500.0)
    assert out["source"] == "heuristic"


# ---------------- the private store ----------------

@pytest.fixture
def store(tmp_path):
    s = ReflectionStore(tmp_path / "reflections.sqlite")
    s.add("run1", "macro", "2016-06-21", "T1",
          {"lesson": "Buying gold into the Brexit vote worked; I sized too small.",
           "what_worked": ["duration hedge"], "what_failed": ["undersized"],
           "tags": ["GOLD", "EVENT_RISK"], "tickers": ["GLD"], "pnl_usd": 320.0,
           "n_trades": 2, "source": "llm"})
    s.add("run1", "quant", "2016-06-21", "T1",
          {"lesson": "Chased a gold breakout and gave it straight back.",
           "tags": ["GOLD"], "tickers": ["GLD"]})
    return s


def test_an_agent_retrieves_its_own_lessons(store):
    hits = store.search("macro", "gold event risk sizing")
    assert hits and "sized too small" in hits[0]["lesson"]


def test_reflections_are_private_to_the_agent_that_wrote_them(store):
    """The whiteboard marks this store Private; that is enforced, not assumed."""
    assert store.search("macro", "chased breakout gave back") == []
    assert store.latest("macro") and all(r["agent_id"] == "macro" for r in store.latest("macro"))
    assert store.count("macro") == 1 and store.count() == 2


def test_lessons_from_today_are_not_recalled_today(store):
    assert store.search("macro", "gold", before_day="2016-06-21") == []
    assert store.search("macro", "gold", before_day="2016-06-22")


def test_structured_fields_survive_the_round_trip(store):
    (hit,) = store.search("macro", "gold")
    assert hit["tags"] == ["GOLD", "EVENT_RISK"]
    assert hit["tickers"] == ["GLD"]
    assert hit["pnl_usd"] == 320.0
    assert hit["source"] == "llm"


def test_an_empty_lesson_is_not_stored(store):
    assert store.add("run1", "macro", "2016-06-22", "T", {"lesson": "   "}) == 0
    assert store.count("macro") == 1


def test_latest_returns_newest_first(store):
    store.add("run1", "macro", "2016-06-22", "T2", {"lesson": "second lesson"})
    assert store.latest("macro", limit=2)[0]["lesson"] == "second lesson"


def test_punctuation_in_a_recall_query_does_not_break_search(store):
    assert store.search("macro", "gold (event-risk): how did I do?!") is not None


def test_runs_can_be_isolated_when_asked(store):
    assert store.search("macro", "gold", run_id="other_run") == []


def test_counts_can_be_scoped_to_one_run(store):
    """The store file outlives a run, so run stats must not include older runs."""
    store.add("run2", "macro", "2016-07-01", "T", {"lesson": "a later run's lesson"})
    assert store.count("macro") == 2
    assert store.count("macro", run_id="run1") == 1
    assert store.count(run_id="run2") == 1


def test_recall_can_be_kept_inside_one_run(store):
    store.add("run2", "macro", "2016-07-01", "T",
              {"lesson": "gold lesson from another run", "tags": ["GOLD"]})
    assert len(store.search("macro", "gold")) == 2
    assert len(store.search("macro", "gold", run_id="run1")) == 1
