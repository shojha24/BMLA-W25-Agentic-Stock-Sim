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

    def spy(digest, state, state_by_agent=None):
        seen.update(state_by_agent or {})
        return original(digest, state, state_by_agent=state_by_agent)

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
