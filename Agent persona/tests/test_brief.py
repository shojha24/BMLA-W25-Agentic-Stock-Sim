import pytest

from agents.town_crier import TownCrierAgent
from data.digest_builder import HeuristicDigestBuilder
from data.news_feed import NewsItem
from sim.actions_db import ActionsDB
from sim.assets_db import AssetsDB
from sim.brief import BriefAssembler, BriefConfig

RUN = "run_1"


@pytest.fixture
def segment():
    items = [NewsItem("n1", "2016-06-24T19:00:00Z", "wire", "Gold surges on safe haven demand",
                      "", ["GLD"])]
    return TownCrierAgent(HeuristicDigestBuilder()).summarize_segment(
        items, "2016-06-24T20:00:00Z", ["GLD", "TLT"])


@pytest.fixture
def state():
    return {"cash_usd": 5000.0,
            "positions": {"GLD": {"qty": 10, "avg_price": 120.0}},
            "prices": {"GLD": 125.0, "TLT": 100.0}}


@pytest.fixture
def dbs(tmp_path):
    actions = ActionsDB(tmp_path / "actions.sqlite")
    assets = AssetsDB(tmp_path / "assets.sqlite")
    fill = lambda agent, ticker, status="FILLED": {
        "agent_id": agent, "timestamp": "2016-06-23T20:00:00Z", "ticker": ticker, "side": "BUY",
        "requested_qty": 10.0, "filled_qty": 10.0 if status != "REJECTED" else 0.0,
        "price": 120.0, "notional": -1200.0, "cost": 0.1, "status": status,
        "reason": "" if status == "FILLED" else "insufficient cash"}
    actions.record(RUN, 1, "2016-06-23", [fill("macro", "GLD"),
                                          fill("macro", "TLT", "REJECTED"),
                                          fill("quant", "QQQ")])
    return actions, assets


def build(dbs, segment, state, **kw):
    actions, assets = dbs
    return BriefAssembler(actions, assets, kw.pop("config", None)).build(
        "macro", segment, state, run_id=RUN, cycle=2, **kw)


def test_brief_has_every_section_from_the_whiteboard(dbs, segment, state):
    brief = build(dbs, segment, state, historical_context="past gold spikes faded")
    for key in ("news_summary", "stocks_discussed", "your_balance", "your_last_trades",
                "historical_context", "your_reflections", "order_instructions"):
        assert key in brief


def test_balance_shows_market_value_and_unrealized_pnl(dbs, segment, state):
    brief = build(dbs, segment, state)
    gld = brief["your_balance"]["positions"]["GLD"]
    assert gld["market_value"] == pytest.approx(1250.0)
    assert gld["unrealized_pnl"] == pytest.approx(50.0)
    assert brief["your_balance"]["equity_usd"] == pytest.approx(6250.0)


def test_last_trades_are_the_agents_own(dbs, segment, state):
    brief = build(dbs, segment, state)
    assert {t["ticker"] for t in brief["your_last_trades"]} == {"GLD", "TLT"}


def test_rejected_orders_are_shown_so_the_agent_can_learn(dbs, segment, state):
    brief = build(dbs, segment, state)
    rejected = [t for t in brief["your_last_trades"] if t["status"] == "REJECTED"]
    assert rejected and rejected[0]["reason"] == "insufficient cash"


def test_peer_actions_are_visible_but_never_the_agents_own(dbs, segment, state):
    brief = build(dbs, segment, state)
    assert brief["peer_recent_actions"]
    assert all(a["agent_id"] != "macro" for a in brief["peer_recent_actions"])


def test_peer_actions_can_be_switched_off(dbs, segment, state):
    brief = build(dbs, segment, state, config=BriefConfig(include_peer_actions=False))
    assert brief["peer_recent_actions"] == []


def test_historical_context_carries_summary_documents_and_questions(dbs, segment, state):
    brief = build(dbs, segment, state, historical_context="summary here",
                  historical_docs=[{"doc_id": "d1", "text": "old news"}])
    ctx = brief["historical_context"]
    assert ctx["summary"] == "summary here"
    assert ctx["documents"][0]["doc_id"] == "d1"
    assert ctx["questions_asked"] == segment.rag_questions["news"]


def test_reflections_slot_is_present_and_empty_until_phase_3(dbs, segment, state):
    assert build(dbs, segment, state)["your_reflections"] == []


def test_order_instructions_are_passed_through(dbs, segment, state):
    brief = build(dbs, segment, state, order_instructions={"max_order_pct_equity": 0.35})
    assert brief["order_instructions"]["max_order_pct_equity"] == 0.35


def test_a_brief_works_without_any_databases(segment, state):
    brief = BriefAssembler().build("macro", segment, state)
    assert brief["your_last_trades"] == [] and brief["your_balance"]["cash_usd"] == 5000.0


def test_brief_computes_what_the_agent_can_actually_trade(dbs, segment, state):
    """The commonest invalid order is selling shares the agent does not hold."""
    brief = build(dbs, segment, state, order_instructions={"max_order_pct_equity": 0.5})
    instructions = brief["order_instructions"]
    assert instructions["you_can_sell_at_most"] == {"GLD": 10}
    # cash 5000, equity 6250, cap 50% -> budget 3125 -> 25 shares of GLD at 125
    assert instructions["you_can_buy_at_most"]["GLD"] == 25
    assert instructions["you_can_buy_at_most"]["TLT"] == 31


def test_nothing_is_sellable_from_an_empty_book(dbs, segment):
    empty = {"cash_usd": 1000.0, "positions": {}, "prices": {"GLD": 125.0}}
    brief = build(dbs, segment, empty)
    assert brief["order_instructions"]["you_can_sell_at_most"] == {}
