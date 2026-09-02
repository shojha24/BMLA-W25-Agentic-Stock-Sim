"""The simulation loop.

One cycle = fetch news -> build digest -> run the panel -> consensus -> score
against the realized move -> rebalance. Backtest mode steps through historical
sessions; live mode does the same thing on a wall-clock interval.

Every cycle is appended to a JSONL run log, so a run can be re-scored later
without re-paying for the LLM calls.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.io import append_jsonl, save_json
from data.market_data import (PriceStore, horizon_to_bars, horizon_to_sessions, is_intraday)
from eval.benchmarks import NaiveForecaster
from eval.metrics import (score_forecasts, summarize_equity, summarize_forecasts,
                          summarize_sim_vs_actual)
from agents.reflection import DayRecord, ReflectionAgent
from agents.town_crier import SegmentBrief, TownCrierAgent
from orchestration.roundtable import Roundtable
from sim.actions_db import ActionsDB
from sim.assets_db import AssetsDB
from sim.brief import BriefAssembler, BriefConfig
from sim.execution import ExecutionConfig, ExecutionVenue, MarketFillVenue, summarize_fills
from sim.portfolio import Portfolio, target_weights
from tools.news_index import LiveNewsIndex
from tools.reflection_store import ReflectionStore

INDEX_TICKER = "SPY"


@dataclass
class EngineConfig:
    universe: List[str]
    horizon: str = "1d"
    initial_cash: float = 100_000.0
    cost_bps: float = 1.0
    max_gross: float = 1.0
    max_name: float = 0.34
    min_confidence: float = 0.10
    allow_short: bool = True
    lookback_hours: float = 24.0
    max_news_items: int = 12
    index_ticker: str = INDEX_TICKER
    # Cycles cut off at the US close (16:00 ET = 20:00 UTC), because the cycle
    # trades at that day's close. Reading news published after the cutoff and
    # filling at the same close would be look-ahead.
    close_hour_utc: int = 20
    # Each agent trades its own book with this much cash (the private Agent Assets DB).
    agent_cash: float = 100_000.0
    trade_agent_books: bool = True
    # Phase 2: Town Crier summarises the segment, retrieval runs once centrally,
    # and each agent reads a 15-Minute Brief instead of a raw digest.
    use_briefs: bool = True
    index_live_news: bool = True
    context_top_k: int = 8
    # Phase 3: end-of-day reflection, stored privately per agent and retrieved
    # back into the brief.
    reflect: bool = True
    reflect_every_cycles: int = 1     # the whiteboard's "2x a day? end of day?" knob
    reflections_top_k: int = 3
    # "run" keeps each run's memory to itself (reproducible ablations);
    # "global" lets agents remember earlier runs in the same store.
    memory_scope: str = "run"


class SimulationEngine:
    def __init__(
        self,
        panel: Roundtable,
        feed,
        digest_builder,
        prices: PriceStore,
        config: EngineConfig,
        benchmarks: Optional[Sequence[NaiveForecaster]] = None,
        log_dir: Optional[Path] = None,
        run_id: str = "",
        venue: Optional[ExecutionVenue] = None,
        execution_config: Optional[ExecutionConfig] = None,
        town_crier: Optional[TownCrierAgent] = None,
        rag_tool: Optional[Any] = None,
        brief_config: Optional[BriefConfig] = None,
        reflection_agent: Optional[ReflectionAgent] = None,
    ):
        self.panel = panel
        self.feed = feed
        self.digest_builder = digest_builder
        self.prices = prices
        self.config = config
        self.benchmarks = list(benchmarks or [])
        self.run_id = run_id or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:4]}"

        root = Path(__file__).resolve().parents[2]      # .../Agent persona
        self.log_dir = Path(log_dir or root / "data" / "runs")
        self.log_path = self.log_dir / f"{self.run_id}.jsonl"

        self.model_names = ["agents_consensus", "agents_round1"] + [b.name for b in self.benchmarks]
        self.portfolios: Dict[str, Portfolio] = {
            name: Portfolio(config.initial_cash, cost_bps=config.cost_bps)
            for name in self.model_names + [f"{config.index_ticker.lower()}_buy_hold"]
        }
        self.curves: Dict[str, List[float]] = {name: [] for name in self.portfolios}
        self.scored: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.model_names}
        self.per_agent_scored: Dict[str, List[Dict[str, Any]]] = {}
        self.revisions: List[Dict[str, Any]] = []
        self.cycles: List[Dict[str, Any]] = []
        self.skipped: List[Dict[str, str]] = []

        # --- Phase 1: each agent trades its own book through a market simulator ---
        self.agent_names = [a.name for a in self.panel.agents]
        self.exec_config = execution_config or ExecutionConfig(
            cost_bps=config.cost_bps, allow_short=config.allow_short)
        self.venue = venue or MarketFillVenue(self.exec_config)
        self.agent_books: Dict[str, Portfolio] = {
            name: Portfolio(config.agent_cash, cost_bps=config.cost_bps)
            for name in self.agent_names
        }
        self.agent_curves: Dict[str, List[float]] = {name: [] for name in self.agent_names}
        self.actions_db = ActionsDB(self.log_dir / "actions.sqlite")       # public
        self.assets_db = AssetsDB(self.log_dir / "agent_assets.sqlite")    # private

        # --- Phase 2: Town Crier, central retrieval, briefs, live news index ---
        self.town_crier = town_crier or TownCrierAgent(digest_builder, use_llm=False)
        self.rag_tool = rag_tool
        self.brief_assembler = BriefAssembler(self.actions_db, self.assets_db, brief_config)
        self.live_index: Optional[LiveNewsIndex] = None
        if config.index_live_news:
            self.live_index = getattr(rag_tool, "live_index", None) or \
                LiveNewsIndex(self.log_dir / "live_news.sqlite")
            if rag_tool is not None and getattr(rag_tool, "live_index", None) is None:
                rag_tool.live_index = self.live_index      # retrieval sees this run's news
        self.segments: List[Dict[str, Any]] = []

        # --- Phase 3: private reflection memory ---
        self.reflection_agent = reflection_agent or ReflectionAgent(use_llm=False)
        self.reflection_store = ReflectionStore(self.log_dir / "reflections.sqlite")
        self.personas = {a.name: getattr(a, "persona", a.name) for a in self.panel.agents}
        self.pending_reflections: Dict[str, DayRecord] = {}
        self.bar_interval = "15m"
        self.reflections_written = 0
        self.fills: List[Dict[str, Any]] = []
        self.cycle_index = 0

    # ---------------- one cycle ----------------

    def _weights(self, forecasts: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        c = self.config
        return target_weights(forecasts, max_gross=c.max_gross, max_name=c.max_name,
                              min_confidence=c.min_confidence, allow_short=c.allow_short)

    def _prices_now(self, day: str, timestamp: str) -> Dict[str, float]:
        """Intraday horizons mark at the bar; daily horizons mark at the close."""
        tickers = self.config.universe + [self.config.index_ticker]
        if is_intraday(self.config.horizon):
            prices = self.prices.prices_at(tickers, timestamp, self.bar_interval)
            if prices:
                return prices
        return self.prices.prices_on(tickers, day)

    def _realized(self, day: str, timestamp: str) -> Dict[str, float]:
        """Forward move over the horizon: bars intraday, sessions otherwise."""
        out: Dict[str, float] = {}
        intraday = is_intraday(self.config.horizon)
        bars = horizon_to_bars(self.config.horizon, 15)
        sessions = horizon_to_sessions(self.config.horizon)
        for ticker in self.config.universe:
            value = (self.prices.forward_return_bps_intraday(ticker, timestamp, bars,
                                                             self.bar_interval)
                     if intraday else
                     self.prices.forward_return_bps(ticker, day, sessions))
            if value is not None:
                out[ticker] = value
        return out

    def run_cycle(self, at: datetime, day: str, score: bool = True) -> Optional[Dict[str, Any]]:
        cfg = self.config
        prices_now = self._prices_now(day, at.strftime("%Y-%m-%dT%H:%M:%SZ"))
        if not prices_now:
            self.skipped.append({"day": day, "reason": "no prices"})
            return None

        # Mark every book before acting, so the curves share a timestamp.
        for name, pf in self.portfolios.items():
            self.curves[name].append(pf.equity(prices_now))

        # The index benchmark buys once and holds.
        idx_name = f"{cfg.index_ticker.lower()}_buy_hold"
        if not self.portfolios[idx_name].positions:
            self.portfolios[idx_name].rebalance_to({cfg.index_ticker: 1.0}, prices_now)

        # The index is a benchmark, not an instrument: keep it out of the price map the
        # agents see, or they will forecast and trade the thing they are measured against.
        universe_prices = {t: px for t, px in prices_now.items() if t in set(cfg.universe)}

        for name, book in self.agent_books.items():
            self.agent_curves[name].append(book.equity(universe_prices))

        news = self.feed.fetch(at, int(cfg.lookback_hours * 60), limit=cfg.max_news_items)
        if not news:
            self.skipped.append({"day": day, "reason": "no news"})
            return None

        self.cycle_index += 1
        timestamp = at.strftime("%Y-%m-%dT%H:%M:%SZ")

        segment = self.town_crier.summarize_segment(news, timestamp, cfg.universe)
        digest = segment.digest
        state = self.portfolios["agents_consensus"].to_state(universe_prices)
        # Each agent sees its own balance sheet: this is the private Agent Assets read.
        state_by_agent = {name: book.to_state(universe_prices)
                          for name, book in self.agent_books.items()}

        brief_by_agent: Optional[Dict[str, Dict[str, Any]]] = None
        context = {"summary": "", "documents": [], "questions": []}
        if cfg.use_briefs:
            # "Historical Context" means news from before this segment: exclude the
            # items the agents are already reading in the digest.
            context = self._retrieve_context(
                segment, day,
                exclude_ids={str(getattr(n, "news_id", "")) for n in news})
            brief_by_agent = {
                name: self.brief_assembler.build(
                    name, segment, state_by_agent[name],
                    run_id=self.run_id, cycle=self.cycle_index,
                    historical_context=context["summary"],
                    historical_docs=context["documents"],
                    reflections=self._recall(name, segment, day),
                    order_instructions={
                        "max_order_pct_equity": self.exec_config.max_order_pct_equity,
                        "shorting_allowed": self.exec_config.allow_short,
                        "whole_shares_only": self.exec_config.whole_shares,
                        "cooldown_cycles": self.exec_config.cooldown_cycles,
                        "cost_bps": self.exec_config.cost_bps,
                        "slippage_bps": self.exec_config.slippage_bps,
                    },
                )
                for name in self.agent_books
            }

        try:
            table = self.panel.run(digest, state, state_by_agent=state_by_agent,
                                   brief_by_agent=brief_by_agent)
        except Exception as exc:
            self.skipped.append({"day": day, "reason": f"panel failed: {type(exc).__name__}: {exc}"})
            return None

        forecasts_by_model: Dict[str, List[Dict[str, Any]]] = {
            "agents_consensus": table["consensus"]["forecasts"],
            "agents_round1": table["consensus_round1"]["forecasts"],
        }
        for bench in self.benchmarks:
            forecasts_by_model[bench.name] = bench.forecast(
                digest=digest, state=state, universe=cfg.universe, day=day,
                prices=self.prices, horizon=cfg.horizon,
            )

        realized = self._realized(day, timestamp) if score else {}
        index_bps = None
        if score:
            index_bps = (self.prices.forward_return_bps_intraday(
                cfg.index_ticker, timestamp, horizon_to_bars(cfg.horizon, 15), self.bar_interval)
                if is_intraday(cfg.horizon) else
                self.prices.forward_return_bps(cfg.index_ticker, day,
                                               horizon_to_sessions(cfg.horizon)))

        cycle_scores: Dict[str, Any] = {}
        for name, forecasts in forecasts_by_model.items():
            if realized:
                rows = score_forecasts(forecasts, realized)
                self.scored[name].extend(rows)
                cycle_scores[name] = summarize_forecasts(rows)
            self.portfolios[name].rebalance_to(self._weights(forecasts), universe_prices)

        # Per-agent scoring: does the consensus beat its own members?
        for out in table["rounds"][-1]["outputs"]:
            key = f"agent:{out['agent_name']}"
            if realized:
                self.per_agent_scored.setdefault(key, []).extend(
                    score_forecasts(out.get("forecasts", []), realized))

        # --- execute each agent's own orders against the market simulator ---
        final_outputs = table["rounds"][-1]["outputs"]
        cycle_fills: List[Dict[str, Any]] = []
        execution: Dict[str, Any] = {}
        if cfg.trade_agent_books:
            orders_by_agent = {out["agent_name"]: out.get("orders", []) or []
                               for out in final_outputs}
            rationales = {f"{out['agent_name']}|{o['ticker']}": o.get("rationale", "")
                          for out in final_outputs for o in (out.get("orders") or [])}
            self.venue.start_cycle()
            cycle_fills = self.venue.execute(orders_by_agent, universe_prices,
                                             self.agent_books, timestamp)
            self.actions_db.record(self.run_id, self.cycle_index, day, cycle_fills, rationales)
            for name, book in self.agent_books.items():
                self.assets_db.snapshot(self.run_id, self.cycle_index, timestamp,
                                        name, book, universe_prices)
            self.fills.extend(cycle_fills)
            execution = summarize_fills(cycle_fills)

        due = (self.cycle_index % max(cfg.reflect_every_cycles, 1)) == 0
        reflections_now = self._settle_reflections(realized, universe_prices) if due else 0
        if due or not self.pending_reflections:
            self._hold_for_reflection(final_outputs, cycle_fills, universe_prices, day, timestamp)

        indexed = self._index_segment(news)

        if table.get("revision"):
            self.revisions.append(table["revision"])

        record = {
            "run_id": self.run_id,
            "timestamp": timestamp,
            "day": day,
            "n_news": len(news),
            "segment": segment.to_dict(),
            "historical_context": context,
            "news_indexed": indexed,
            "reflections_written": reflections_now,
            "reflections_recalled": {
                name: len((brief_by_agent or {}).get(name, {}).get("your_reflections", []))
                for name in self.agent_books
            } if brief_by_agent else {},
            "digest": digest,
            "prices": prices_now,
            "consensus": table["consensus"],
            "consensus_round1": table["consensus_round1"],
            "agent_outputs": table["rounds"][-1]["outputs"],
            "revision": table.get("revision", {}),
            "errors": table.get("errors", []),
            "realized_bps": {k: round(v, 2) for k, v in realized.items()},
            "index_bps": None if index_bps is None else round(index_bps, 2),
            "cycle_scores": cycle_scores,
            "books": {name: pf.snapshot(prices_now) for name, pf in self.portfolios.items()},
            "agent_books": {name: book.snapshot(universe_prices)
                            for name, book in self.agent_books.items()},
            "fills": cycle_fills,
            "execution": execution,
        }
        append_jsonl(self.log_path, record)
        self.cycles.append({"day": day, "n_news": len(news), "consensus": table["consensus"],
                            "cycle_scores": cycle_scores, "realized_bps": realized,
                            "index_bps": index_bps})
        self.segments.append({"day": day, "source": segment.source, "n_items": segment.n_items,
                              "context_docs": len(context["documents"])})
        return record

    def _retrieve_context(self, segment: SegmentBrief, day: str,
                          exclude_ids: Optional[set] = None) -> Dict[str, Any]:
        """Doc Retrieval, run once for the desk on the Town Crier's questions."""
        questions = [q for q in segment.rag_questions.get("news", []) if q.strip()]
        if self.rag_tool is None or not questions:
            return {"summary": "", "documents": [], "questions": questions}

        seen, docs = set(exclude_ids or ()), []
        per_question = max(2, self.config.context_top_k // max(len(questions), 1))
        for question in questions[:4]:
            rows, _ = self.rag_tool.retrieve(
                query=question, top_k=per_question,
                stock_filter=list(self.config.universe), cutoff_date=day)
            for row in rows:
                if row["doc_id"] in seen:
                    continue
                seen.add(row["doc_id"])
                docs.append(row)
            if len(docs) >= self.config.context_top_k:
                break

        docs = docs[: self.config.context_top_k]
        summary = self.town_crier.summarize_context(docs, "; ".join(questions[:2]))
        return {"summary": summary, "documents": docs, "questions": questions}

    def _recall(self, agent_name: str, segment: SegmentBrief, day: str) -> List[Dict[str, Any]]:
        """Vector Store 2 read: this agent's own lessons from similar setups."""
        if not self.config.reflect:
            return []
        questions = segment.rag_questions.get("insights") or []
        query = " ".join(questions) or segment.summary
        scope = self.run_id if self.config.memory_scope == "run" else None
        found = self.reflection_store.search(
            agent_name, query, top_k=self.config.reflections_top_k, before_day=day,
            run_id=scope)
        if not found:
            # Nothing matched the wording; fall back to the most recent lessons.
            found = self.reflection_store.latest(
                agent_name, limit=self.config.reflections_top_k, before_day=day, run_id=scope)
        return [{k: r[k] for k in ("day", "lesson", "what_worked", "what_failed",
                                   "tags", "tickers", "pnl_usd")} for r in found]

    def _hold_for_reflection(self, outputs: Sequence[Dict[str, Any]],
                             fills: Sequence[Dict[str, Any]], prices: Dict[str, float],
                             day: str, timestamp: str) -> None:
        """Park this cycle's activity until the market has judged it."""
        if not self.config.reflect:
            return
        for out in outputs:
            name = out["agent_name"]
            book = self.agent_books.get(name)
            if book is None:
                continue
            self.pending_reflections[name] = DayRecord(
                agent_id=name, persona=self.personas.get(name, name), day=day,
                timestamp=timestamp,
                trades=[f for f in fills if f["agent_id"] == name],
                forecasts=list(out.get("forecasts") or []),
                equity=book.equity(prices),
                positions={t: dict(p) for t, p in book.positions.items()},
            )

    def _settle_reflections(self, realized_bps: Dict[str, float],
                            prices: Dict[str, float]) -> int:
        """Write reflections for the previous cycle, now that its outcome is known.

        Reflecting a cycle late is deliberate: judging a day's trades needs the
        move that followed them, which did not exist when they were placed.
        """
        if not self.config.reflect or not self.pending_reflections:
            return 0
        written = 0
        for name, record in list(self.pending_reflections.items()):
            book = self.agent_books.get(name)
            if book is None:
                continue
            reflection = self.reflection_agent.reflect(record, realized_bps, book.equity(prices))
            if reflection:
                self.reflection_store.add(self.run_id, name, record.day, record.timestamp,
                                          reflection)
                written += 1
        self.pending_reflections.clear()
        self.reflections_written += written
        return written

    def _index_segment(self, news: Sequence[Any]) -> int:
        """Index after the cycle has read its context, so this segment is 'old news'
        from the next cycle onward and never retrieves itself."""
        return self.live_index.add(news) if self.live_index is not None else 0

    # ---------------- drivers ----------------

    def run_backtest(
        self,
        start: str,
        end: str,
        step_days: int = 1,
        max_cycles: Optional[int] = None,
        on_cycle: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        self.prices.ensure(list(dict.fromkeys(cfg.universe + [cfg.index_ticker])), start, end)

        # A symbol with no price history cannot be scored or traded; drop it and say so.
        missing = [t for t in cfg.universe if not self.prices.has_data(t)]
        if missing:
            cfg.universe = [t for t in cfg.universe if t not in missing]
            self.skipped.append({"day": "-", "reason": f"no price data: {','.join(missing)}"})
        if not cfg.universe:
            raise RuntimeError("No universe ticker has price data.")

        sessions = self.prices.sessions(cfg.index_ticker, start, end)
        if not sessions:
            raise RuntimeError(f"No trading sessions between {start} and {end} for {cfg.index_ticker}")

        horizon_sessions = horizon_to_sessions(cfg.horizon)
        tradeable = sessions[:-horizon_sessions] if horizon_sessions < len(sessions) else []
        chosen = tradeable[::step_days]
        if max_cycles:
            chosen = chosen[:max_cycles]

        done = 0
        for day in chosen:
            at = datetime.strptime(day, "%Y-%m-%d").replace(hour=cfg.close_hour_utc, tzinfo=timezone.utc)
            record = self.run_cycle(at, day, score=True)
            if record and on_cycle:
                done += 1
                on_cycle(done, day, record)

        last_day = sessions[-1]
        final_prices = self.prices.prices_on(cfg.universe + [cfg.index_ticker], last_day)
        for name, pf in self.portfolios.items():
            self.curves[name].append(pf.equity(final_prices))
        for name, book in self.agent_books.items():
            self.agent_curves[name].append(book.equity(final_prices))

        # Settle the last cycle's reflection now that its outcome is on the tape.
        if self.pending_reflections:
            settle_day = self.cycles[-1]["day"] if self.cycles else last_day
            sessions_ahead = horizon_to_sessions(cfg.horizon)
            last_realized = {}
            for ticker in cfg.universe:
                value = self.prices.forward_return_bps(ticker, settle_day, sessions_ahead)
                if value is not None:
                    last_realized[ticker] = value
            self._settle_reflections(last_realized, final_prices)

        return self.report(mode="backtest", window={"start": start, "end": end,
                                                    "cycles": len(self.cycles),
                                                    "sessions": len(sessions)})

    def run_live(self, interval_minutes: int = 15, cycles: int = 1,
                 on_cycle: Optional[Callable[[int, str, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """Poll the live feed every `interval_minutes`.

        Forecasts cannot be scored at the moment they are made; the run log
        carries them so a later pass can score them once prices exist.
        """
        cfg = self.config
        for i in range(cycles):
            now = datetime.now(timezone.utc)
            day = self.prices.sessions(cfg.index_ticker)[-1] if self.prices.load(cfg.index_ticker) else now.strftime("%Y-%m-%d")
            record = self.run_cycle(now, day, score=False)
            if record and on_cycle:
                on_cycle(i + 1, day, record)
            if i < cycles - 1:
                time.sleep(interval_minutes * 60)
        return self.report(mode="live", window={"cycles": len(self.cycles),
                                                "interval_minutes": interval_minutes})

    # ---------------- reporting ----------------

    def report(self, mode: str, window: Dict[str, Any]) -> Dict[str, Any]:
        forecast_stats = {name: summarize_forecasts(rows) for name, rows in self.scored.items() if rows}
        forecast_stats.update({name: summarize_forecasts(rows)
                               for name, rows in self.per_agent_scored.items() if rows})
        equity_stats = {name: summarize_equity(curve) for name, curve in self.curves.items() if len(curve) > 1}
        # Per-agent books are normalised to the same starting equity as the consensus
        # book so the two columns are comparable in one table.
        scale = self.config.initial_cash / self.config.agent_cash if self.config.agent_cash else 1.0
        equity_stats.update({
            f"book:{name}": summarize_equity([v * scale for v in curve])
            for name, curve in self.agent_curves.items() if len(curve) > 1
        })

        revision = {}
        if self.revisions:
            revision = {
                "cycles_with_revision": len(self.revisions),
                "mean_flip_rate": round(sum(r.get("flip_rate", 0.0) for r in self.revisions) / len(self.revisions), 4),
                "mean_abs_delta_bps": round(sum(r.get("mean_abs_delta_bps", 0.0) for r in self.revisions) / len(self.revisions), 2),
                "mean_delta_confidence": round(sum(r.get("mean_delta_confidence", 0.0) for r in self.revisions) / len(self.revisions), 4),
            }

        report = {
            "run_id": self.run_id,
            "mode": mode,
            "window": window,
            "config": {
                "universe": self.config.universe, "horizon": self.config.horizon,
                "cost_bps": self.config.cost_bps, "max_gross": self.config.max_gross,
                "allow_short": self.config.allow_short, "initial_cash": self.config.initial_cash,
            },
            "feed": self.feed.describe() if hasattr(self.feed, "describe") else {},
            "digest_builder": getattr(self.digest_builder, "name", "unknown"),
            "reflection": {
                "enabled": self.config.reflect,
                "reflector": "llm" if self.reflection_agent.use_llm else "heuristic",
                "every_cycles": self.config.reflect_every_cycles,
                "written": self.reflections_written,
                "scope": self.config.memory_scope,
                "per_agent": {name: self.reflection_store.count(name, run_id=self.run_id)
                              for name in self.agent_books},
                "store": str(self.reflection_store.db_path),
            },
            "briefing": {
                "enabled": self.config.use_briefs,
                "town_crier": "llm" if self.town_crier.use_llm else "heuristic",
                "segments": len(self.segments),
                "mean_context_docs": round(
                    sum(s["context_docs"] for s in self.segments) / len(self.segments), 2)
                if self.segments else 0.0,
                "live_index_rows": self.live_index.count() if self.live_index is not None else 0,
            },
            "agents": self.panel.agents_names if hasattr(self.panel, "agents_names") else [a.name for a in self.panel.agents],
            "forecast_metrics": forecast_stats,
            "sim_vs_actual": summarize_sim_vs_actual(self.cycles),
            "portfolio_metrics": equity_stats,
            "equity_curves": {k: [round(v, 2) for v in c] for k, c in self.curves.items()},
            "revision": revision,
            "execution": {
                "venue": getattr(self.venue, "name", "none"),
                "config": vars(self.exec_config),
                "totals": summarize_fills(self.fills),
                "per_agent": self.actions_db.summary(self.run_id),
            },
            "agent_equity_curves": {k: [round(v, 2) for v in c] for k, c in self.agent_curves.items()},
            "skipped": self.skipped,
            "log_path": str(self.log_path),
            "actions_db": str(self.actions_db.db_path),
            "assets_db": str(self.assets_db.db_path),
        }
        save_json(self.log_dir / f"{self.run_id}_report.json", report)
        return report
