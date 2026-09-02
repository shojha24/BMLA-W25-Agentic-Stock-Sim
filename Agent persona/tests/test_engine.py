"""End-to-end simulation on synthetic prices and fixture news - no network."""
import json
from pathlib import Path

import pytest

from agents.persona import LLMPersonaAgent
from agents.personas import PERSONAS
from data.digest_builder import HeuristicDigestBuilder
from data.market_data import PriceStore
from data.news_feed import FixtureNewsFeed
from eval.benchmarks import build_benchmarks
from llm.mock_client import MockChatClient
from orchestration.roundtable import Roundtable
from sim.engine import EngineConfig, SimulationEngine

UNIVERSE = ["AAA", "BBB"]


class _NoRag:
    def retrieve(self, *a, **k):
        return [], ""

    def status(self):
        return {"mode": "disabled"}


@pytest.fixture
def prices(tmp_path):
    store = PriceStore(cache_dir=tmp_path / "prices")
    days = [f"2016-01-{d:02d}" for d in range(4, 20)]
    for ticker, drift in (("AAA", 1.01), ("BBB", 0.995), ("SPY", 1.002)):
        px, rows = 100.0, []
        for day in days:
            rows.append(f"{day},{px:.4f}")
            px *= drift
        (tmp_path / "prices" / f"{ticker}.csv").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "prices" / f"{ticker}.csv").write_text("date,close\n" + "\n".join(rows) + "\n")
    return store


@pytest.fixture
def feed(tmp_path):
    path = tmp_path / "digest.json"
    path.write_text(json.dumps({"timestamp": "2016-01-04T21:00:00Z", "news_digest": [
        {"news_id": "n1", "headline": "AAA surges on strong guidance and upgrades",
         "summary": "beats estimates", "tickers_mentioned": ["AAA"]},
        {"news_id": "n2", "headline": "BBB plunges on weak demand warning",
         "summary": "misses", "tickers_mentioned": ["BBB"]},
    ]}))
    return FixtureNewsFeed(path)


def make_engine(prices, feed, tmp_path, rounds=2, benchmarks=None):
    client = MockChatClient()
    panel = Roundtable(
        [LLMPersonaAgent(spec=PERSONAS[k], client=client, model="mock", rag_tool=_NoRag())
         for k in ("macro_econ", "quant_momentum")],
        rounds=rounds,
    )
    config = EngineConfig(universe=UNIVERSE, initial_cash=10000.0, cost_bps=1.0, index_ticker="SPY")
    return SimulationEngine(panel, feed, HeuristicDigestBuilder(), prices, config,
                            benchmarks=build_benchmarks(benchmarks if benchmarks is not None
                                                        else ["always_long", "persistence"]),
                            log_dir=tmp_path / "runs")


def test_each_agent_trades_its_own_book(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    report = engine.run_backtest("2016-01-04", "2016-01-19")

    assert set(engine.agent_books) == {"macro_econ_llm_v2", "quant_momentum_llm_v1"}
    for name in engine.agent_books:
        assert f"book:{name}" in report["portfolio_metrics"]
        assert len(report["agent_equity_curves"][name]) > 2
    # separate books: one agent's cash cannot be spent by another
    equities = [b.equity({"AAA": 100.0, "BBB": 100.0}) for b in engine.agent_books.values()]
    assert all(e > 0 for e in equities)


def test_agents_receive_their_own_balance_sheet(prices, feed, tmp_path):
    """The private Agent Assets read: each agent sees its own cash, not a shared book."""
    engine = make_engine(prices, feed, tmp_path)
    seen = {}

    original = engine.panel.run

    def spy(digest, state, state_by_agent=None, **kwargs):
        seen.update(state_by_agent or {})
        return original(digest, state, state_by_agent=state_by_agent, **kwargs)

    engine.panel.run = spy
    engine.run_backtest("2016-01-04", "2016-01-12")
    assert set(seen) == set(engine.agent_books)
    for name, agent_state in seen.items():
        assert "cash_usd" in agent_state and "positions" in agent_state
        assert "SPY" not in agent_state["prices"]


def test_orders_are_executed_and_recorded_in_the_actions_db(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    report = engine.run_backtest("2016-01-04", "2016-01-19")

    totals = report["execution"]["totals"]
    assert totals["n_orders"] > 0
    assert report["execution"]["venue"] == "market_fill"

    rows = engine.actions_db.last_trades(engine.run_id, "macro_econ_llm_v2", limit=50)
    assert rows
    assert all(r["status"] != "REJECTED" for r in rows)
    assert engine.assets_db.latest(engine.run_id, "macro_econ_llm_v2")["equity"] > 0


def test_execution_can_be_switched_off(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    engine.config.trade_agent_books = False
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    assert report["execution"]["totals"] == {"n_orders": 0}
    assert report["forecast_metrics"]["agents_consensus"]["n_forecasts"] > 0


def test_backtest_produces_scored_forecasts_and_curves(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    report = engine.run_backtest("2016-01-04", "2016-01-19")

    assert report["window"]["cycles"] > 5
    assert "agents_consensus" in report["forecast_metrics"]
    assert "always_long" in report["forecast_metrics"]
    assert report["portfolio_metrics"]["spy_buy_hold"]["cumulative_return"] > 0
    assert len(report["equity_curves"]["agents_consensus"]) == report["window"]["cycles"] + 1


def test_every_cycle_is_logged_as_jsonl(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    lines = Path(report["log_path"]).read_text().strip().splitlines()
    assert len(lines) == report["window"]["cycles"]
    first = json.loads(lines[0])
    assert first["consensus"]["forecasts"] and first["realized_bps"] and first["books"]


def test_always_long_beats_a_short_only_view_on_a_rising_ticker(prices, feed, tmp_path):
    """Sanity: the scorer rewards being right about direction."""
    engine = make_engine(prices, feed, tmp_path, benchmarks=["always_long"])
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    scored = [s for s in engine.scored["always_long"] if s["ticker"] == "AAA"]
    assert all(s["sign_correct"] for s in scored)          # AAA rises every session
    assert report["forecast_metrics"]["always_long"]["hit_rate"] == 0.5   # AAA up, BBB down


def test_agents_never_see_the_benchmark_ticker(prices, feed, tmp_path):
    """SPY is the yardstick; letting the panel trade it would flatter the comparison."""
    import json as _json
    engine = make_engine(prices, feed, tmp_path)
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    first = _json.loads(Path(report["log_path"]).read_text().splitlines()[0])
    agent_state_tickers = {f["ticker"] for f in first["consensus"]["forecasts"]}
    assert "SPY" not in agent_state_tickers
    assert "SPY" not in first["books"]["agents_consensus"]["weights"]
    assert first["books"]["spy_buy_hold"]["weights"].get("SPY", 0) > 0


def test_round1_consensus_is_tracked_separately(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    assert "agents_round1" in report["forecast_metrics"]
    assert report["revision"]["cycles_with_revision"] > 0


def test_single_round_panel_skips_the_revision_stats(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path, rounds=1)
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    assert report["revision"] == {}


def test_a_ticker_with_no_price_data_is_dropped_not_fatal(prices, feed, tmp_path, monkeypatch):
    def dead(*a, **k):
        raise RuntimeError("unknown symbol")
    monkeypatch.setattr(prices, "download", dead)     # keeps the test offline

    engine = make_engine(prices, feed, tmp_path)
    engine.config.universe = ["AAA", "ZZZ"]           # ZZZ has no price history
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    assert report["window"]["cycles"] > 0
    assert report["config"]["universe"] == ["AAA"]
    assert any("no price data" in s["reason"] for s in report["skipped"])


def test_cli_execution_flags_reach_the_venue():
    """The CLI once built an ExecutionConfig and dropped it on the floor."""
    import simulate

    args = simulate.build_parser().parse_args([
        "backtest", "--mode", "mock", "--cooldown-cycles", "3", "--slippage-bps", "7",
        "--max-order-pct", "0.2", "--agent-cash", "50000",
    ])
    engine = simulate.make_engine(args, ["QQQ", "TLT"])
    cfg = engine.venue.config
    assert (cfg.cooldown_cycles, cfg.slippage_bps, cfg.max_order_pct_equity) == (3, 7.0, 0.2)
    assert cfg.allow_short is False
    assert all(book.cash == 50000.0 for book in engine.agent_books.values())


def test_agents_are_handed_a_brief_not_a_raw_digest(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    seen = {}
    original = engine.panel.run

    def spy(digest, state, state_by_agent=None, brief_by_agent=None, **kw):
        seen.update(brief_by_agent or {})
        return original(digest, state, state_by_agent=state_by_agent,
                        brief_by_agent=brief_by_agent, **kw)

    engine.panel.run = spy
    engine.run_backtest("2016-01-04", "2016-01-19")

    assert set(seen) == set(engine.agent_books)
    brief = seen["macro_econ_llm_v2"]
    assert brief["news_summary"]
    assert brief["your_balance"]["cash_usd"] > 0
    assert "your_reflections" in brief
    assert brief["order_instructions"]["max_order_pct_equity"] > 0


def test_briefs_can_be_disabled_for_ablation(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    engine.config.use_briefs = False
    seen = {"brief": "unset"}
    original = engine.panel.run

    def spy(digest, state, state_by_agent=None, brief_by_agent=None, **kw):
        seen["brief"] = brief_by_agent
        return original(digest, state, state_by_agent=state_by_agent,
                        brief_by_agent=brief_by_agent, **kw)

    engine.panel.run = spy
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    assert seen["brief"] is None
    assert report["briefing"]["enabled"] is False


def test_this_runs_news_is_indexed_for_later_retrieval(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    assert engine.live_index is not None
    assert report["briefing"]["live_index_rows"] > 0


def test_historical_context_never_returns_the_current_segment(prices, feed, tmp_path):
    """Indexing happens after retrieval, so a cycle cannot cite itself as history."""
    import json as _json
    engine = make_engine(prices, feed, tmp_path)
    engine.rag_tool = type("R", (), {"retrieve": staticmethod(
        lambda **kw: ([{"doc_id": "n1", "date": "2016-01-01", "text": "old", "stocks": []}], ""))})()
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    for line in Path(report["log_path"]).read_text().splitlines():
        record = _json.loads(line)
        segment_ids = {i["news_id"] for i in record["digest"]["news_digest"]}
        context_ids = {d["doc_id"] for d in record["historical_context"]["documents"]}
        assert not (segment_ids & context_ids)


def test_reflections_are_written_and_recalled_into_later_briefs(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    briefs = []
    original = engine.panel.run

    def spy(digest, state, state_by_agent=None, brief_by_agent=None, **kw):
        briefs.append(brief_by_agent or {})
        return original(digest, state, state_by_agent=state_by_agent,
                        brief_by_agent=brief_by_agent, **kw)

    engine.panel.run = spy
    report = engine.run_backtest("2016-01-04", "2016-01-19")

    assert report["reflection"]["written"] > 0
    assert engine.reflection_store.count("macro_econ_llm_v2") > 0
    # the loop closes: nothing to recall at the start, lessons later
    assert briefs[0]["macro_econ_llm_v2"]["your_reflections"] == []
    assert any(b["macro_econ_llm_v2"]["your_reflections"] for b in briefs[2:])


def test_an_agent_never_sees_another_agents_reflections(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    briefs = []
    original = engine.panel.run

    def spy(digest, state, state_by_agent=None, brief_by_agent=None, **kw):
        briefs.append(brief_by_agent or {})
        return original(digest, state, state_by_agent=state_by_agent,
                        brief_by_agent=brief_by_agent, **kw)

    engine.panel.run = spy
    engine.run_backtest("2016-01-04", "2016-01-19")

    for brief_set in briefs:
        for agent, brief in brief_set.items():
            for lesson in brief["your_reflections"]:
                stored = engine.reflection_store.latest(agent, limit=50)
                assert lesson["lesson"] in {s["lesson"] for s in stored}


def test_a_reflection_only_uses_outcomes_that_already_happened(prices, feed, tmp_path):
    """Reflection is deferred one cycle: a day is judged after the market moves."""
    engine = make_engine(prices, feed, tmp_path)
    engine.run_backtest("2016-01-04", "2016-01-19")
    days = [r["day"] for r in engine.reflection_store.latest("macro_econ_llm_v2", limit=50)]
    cycle_days = [c["day"] for c in engine.cycles]
    assert days
    # every reflected day is a cycle that already ran
    assert set(days).issubset(set(cycle_days))


def test_reflection_can_be_switched_off(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    engine.config.reflect = False
    report = engine.run_backtest("2016-01-04", "2016-01-19")
    assert report["reflection"]["written"] == 0
    assert engine.reflection_store.count() == 0


def test_a_run_can_be_resumed_where_it_stopped(prices, feed, tmp_path):
    """A 15-minute loop meant to run for days must survive a restart."""
    first = make_engine(prices, feed, tmp_path)
    first.run_backtest("2016-01-04", "2016-01-11")
    run_id = first.run_id
    marks = {"AAA": 100.0, "BBB": 100.0}
    before = {name: (book.equity(marks), dict(book.positions))
              for name, book in first.agent_books.items()}
    cycles_before = first.cycle_index
    assert cycles_before > 0

    second = make_engine(prices, feed, tmp_path)
    state = second.resume(run_id)

    assert second.run_id == run_id
    assert state["cycles_before"] == cycles_before
    for name, book in second.agent_books.items():
        equity, positions = before[name]
        assert book.equity(marks) == pytest.approx(equity)
        # positions come back too, not just cash (whatever they were, flat included)
        assert {t: round(p["qty"], 4) for t, p in book.positions.items()} == \
               {t: round(p["qty"], 4) for t, p in positions.items()}


def test_resuming_continues_the_cycle_count_and_the_curves(prices, feed, tmp_path):
    first = make_engine(prices, feed, tmp_path)
    first.run_backtest("2016-01-04", "2016-01-11")

    second = make_engine(prices, feed, tmp_path)
    second.resume(first.run_id)
    seeded = len(second.agent_curves["macro_econ_llm_v2"])
    report = second.run_backtest("2016-01-12", "2016-01-19")

    assert seeded > 0
    assert second.cycle_index > first.cycle_index
    assert len(report["agent_equity_curves"]["macro_econ_llm_v2"]) > seeded
    assert report["resumed_from"]["run_id"] == first.run_id


def test_a_resumed_run_keeps_its_memory_and_trade_history(prices, feed, tmp_path):
    first = make_engine(prices, feed, tmp_path)
    first.run_backtest("2016-01-04", "2016-01-11")
    reflections_before = first.reflection_store.count(run_id=first.run_id)

    second = make_engine(prices, feed, tmp_path)
    state = second.resume(first.run_id)
    assert sum(state["reflections"].values()) == reflections_before
    assert sum(state["prior_trades"].values()) >= 0

    second.run_backtest("2016-01-12", "2016-01-19")
    assert second.reflection_store.count(run_id=first.run_id) > reflections_before


def test_resuming_an_unknown_run_starts_clean(prices, feed, tmp_path):
    engine = make_engine(prices, feed, tmp_path)
    state = engine.resume("run_that_never_happened")
    assert state["books"] == {}
    assert state["cycles_before"] == 0
    assert all(book.cash == engine.config.agent_cash for book in engine.agent_books.values())
