"""Backtest, live loop, and ablations.

  # replay 2016 H1 from the local archive, no API key needed
  python "Agent persona/src/simulate.py" backtest --start 2016-01-04 --end 2016-06-30 --mode mock

  # the real panel on live headlines, every 15 minutes
  python "Agent persona/src/simulate.py" live --interval 15 --cycles 4 --mode llm

  # is RAG / the second round / the third agent earning its keep?
  python "Agent persona/src/simulate.py" ablate --start 2016-01-04 --end 2016-03-31 --mode mock
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.baseline import SentimentBaselineAgent          # noqa: E402
from agents.reflection import ReflectionAgent               # noqa: E402
from agents.town_crier import TownCrierAgent                # noqa: E402
from agents.persona import LLMPersonaAgent                  # noqa: E402
from agents.personas import DEFAULT_PANEL, PERSONAS         # noqa: E402
from core.io import load_env, save_json                     # noqa: E402
from data.digest_builder import build_digest_builder        # noqa: E402
from data.market_data import PriceStore                     # noqa: E402
from data.news_feed import build_feed                       # noqa: E402
from eval.benchmarks import DEFAULT_BENCHMARKS, build_benchmarks  # noqa: E402
from eval.metrics import compare                            # noqa: E402
from eval.rescore import rescore_run                        # noqa: E402
from orchestration.roundtable import Roundtable             # noqa: E402
from sim.engine import EngineConfig, SimulationEngine       # noqa: E402
from sim.execution import ExecutionConfig, MarketFillVenue  # noqa: E402
from tools.rag import RAGNewsTool                           # noqa: E402

# Tickers with real coverage in the local archive (SPY and UUP have none).
DEFAULT_UNIVERSE = "QQQ,TLT,GLD,XLE,IWM,NVDA"


def make_client(mode: str, horizon: str):
    if mode == "mock":
        from llm.mock_client import MockChatClient
        return MockChatClient(horizon=horizon)
    from llm.openrouter_client import OpenRouterClient
    return OpenRouterClient(api_key=os.getenv("OPENROUTER_API_KEY"),
                            site_url="http://localhost", app_title="Agentic Stock Sim")


def make_panel(args, rag_tool, personas=None, rounds=None, use_rag=None):
    horizon = args.horizon
    if args.mode == "hardcoded":
        return Roundtable([SentimentBaselineAgent(horizon=horizon)], rounds=1, horizon=horizon)

    client = make_client(args.mode, horizon)
    keys = personas if personas is not None else [k.strip() for k in args.agents.split(",") if k.strip()]
    unknown = [k for k in keys if k not in PERSONAS]
    if unknown:
        raise SystemExit(f"Unknown persona(s): {unknown}. Available: {sorted(PERSONAS)}")

    import dataclasses
    agents = []
    for key in keys:
        spec = PERSONAS[key]
        if horizon != spec.default_horizon:
            spec = dataclasses.replace(spec, default_horizon=horizon)
        agents.append(LLMPersonaAgent(spec=spec, client=client, model=args.model,
                                      rag_tool=rag_tool,
                                      use_rag=(not args.no_rag) if use_rag is None else use_rag,
                                      max_order_pct_equity=args.max_order_pct))
    return Roundtable(agents, rounds=rounds if rounds is not None else args.rounds, horizon=horizon)


def make_engine(args, universe, panel=None, rag_tool=None, run_id="", benchmarks=None):
    run_id = run_id or getattr(args, "resume", "") or ""
    rag_tool = rag_tool or RAGNewsTool()
    panel = panel or make_panel(args, rag_tool)
    feed = build_feed(args.feed, universe,
                      fixture_path=Path(args.fixture) if getattr(args, "fixture", "") else None)
    digest_client = None if args.digest_mode != "llm" else make_client(args.mode, args.horizon)
    builder = build_digest_builder(args.digest_mode, digest_client, args.model)
    config = EngineConfig(
        universe=list(universe), horizon=args.horizon, initial_cash=args.cash,
        cost_bps=args.cost_bps, max_gross=args.max_gross, allow_short=not args.long_only,
        lookback_hours=args.lookback_hours, agent_cash=args.agent_cash,
        trade_agent_books=not args.no_agent_books, use_briefs=not args.no_briefs,
        index_live_news=not args.no_live_index, context_top_k=args.context_top_k,
        reflect=not args.no_reflections, reflections_top_k=args.reflections_top_k,
        memory_scope=args.memory_scope, reflect_every_cycles=args.reflect_every,
    )

    # The Town Crier writes the segment summary and the retrieval questions. It uses
    # the LLM whenever the panel does, unless told otherwise.
    crier_mode = args.town_crier
    if crier_mode == "auto":
        crier_mode = "llm" if args.mode == "llm" else "heuristic"
    crier_client = make_client(args.mode, args.horizon) if crier_mode == "llm" else None
    town_crier = TownCrierAgent(builder, client=crier_client, model=args.model,
                                use_llm=crier_mode == "llm",
                                max_context_docs=args.context_top_k)

    # End-of-day reflection: same default policy as the Town Crier.
    reflect_mode = args.reflection_mode
    if reflect_mode == "auto":
        reflect_mode = "llm" if args.mode == "llm" else "heuristic"
    reflection_agent = ReflectionAgent(
        client=make_client(args.mode, args.horizon) if reflect_mode == "llm" else None,
        model=args.model, use_llm=reflect_mode == "llm")
    execution_config = ExecutionConfig(
        cost_bps=args.cost_bps, slippage_bps=args.slippage_bps,
        allow_short=args.agents_may_short,      # off by default: buy/sell of held shares only
        max_order_pct_equity=args.max_order_pct, cooldown_cycles=args.cooldown_cycles,
        max_position_pct_equity=args.max_position_pct,
    )
    names = None if args.benchmarks == "default" else (
        [] if args.benchmarks in ("", "none") else args.benchmarks.split(","))
    return SimulationEngine(panel, feed, builder, PriceStore(), config,
                            benchmarks=benchmarks if benchmarks is not None else build_benchmarks(names),
                            run_id=run_id,
                            venue=MarketFillVenue(execution_config),
                            execution_config=execution_config,
                            town_crier=town_crier, rag_tool=rag_tool,
                            reflection_agent=reflection_agent)


def resume_engine(args, engine) -> None:
    """Pick a previous run back up, if asked."""
    run_id = getattr(args, "resume", "")
    if not run_id:
        return
    state = engine.resume(run_id)
    if not state["books"]:
        print(f"[warn] nothing stored for run '{run_id}' - starting fresh under that id. "
              f"Known runs: {', '.join(engine.assets_db.books_in_run(run_id)) or 'none'}")
        return
    books = ", ".join(f"{k.replace('_llm_v1', '').replace('_llm_v2', '')} ${v:,.0f}"
                      for k, v in state["books"].items() if k in engine.agent_books)
    print(f"resumed {run_id}: {state['cycles_before']} cycles already run | books {books} | "
          f"{sum(state['reflections'].values())} reflections, "
          f"{sum(state['prior_trades'].values())} prior trades carried over")


def print_report(report: dict) -> None:
    print(f"\n=== {report['mode']} {report['run_id']} ===")
    print(f"universe {','.join(report['config']['universe'])} | horizon {report['config']['horizon']} "
          f"| digest {report['digest_builder']} | agents {','.join(report['agents'])}")
    print(f"window {report['window']}")

    fm = report["forecast_metrics"]
    if fm:
        print(f"\n{'model':28} {'n':>5} {'hit':>7} {'wtd hit':>8} {'brier':>7} {'MAE bps':>8} {'flat%':>6}")
        for name, m in sorted(fm.items(), key=lambda kv: -(kv[1].get("hit_rate") or 0)):
            print(f"{name:28} {m['n_directional']:5d} "
                  f"{(m['hit_rate'] or 0):7.3f} {(m['confidence_weighted_hit_rate'] or 0):8.3f} "
                  f"{(m['brier_score'] or 0):7.3f} {(m['mean_abs_error_bps'] or 0):8.1f} "
                  f"{(m['flat_share'] or 0) * 100:5.0f}%")

    pm = report["portfolio_metrics"]
    if pm:
        print(f"\n{'model':28} {'return':>9} {'ann ret':>9} {'sharpe':>8} {'max dd':>8}")
        for row in compare(pm, key="sharpe"):
            ann = row.get("annualized_return")
            ann_txt = f"{ann * 100:8.2f}%" if ann is not None else f"{'-':>9}"
            print(f"{row['model']:28} {row['cumulative_return'] * 100:8.2f}% {ann_txt} "
                  f"{(row.get('sharpe') or 0):8.2f} {(row.get('max_drawdown') or 0) * 100:7.1f}%")

    br = report.get("briefing", {})
    if br:
        print(f"\nbriefing: {'on' if br.get('enabled') else 'off'} | town crier {br.get('town_crier')} "
              f"| {br.get('mean_context_docs')} context docs/cycle "
              f"| live index {br.get('live_index_rows')} rows")

    rf = report.get("reflection", {})
    if rf:
        per_agent = ", ".join(f"{k.split('_llm')[0]} {v}" for k, v in rf.get("per_agent", {}).items())
        print(f"reflections: {'on' if rf.get('enabled') else 'off'} | {rf.get('reflector')} "
              f"| {rf.get('written')} written ({per_agent})")

    ex = report.get("execution", {})
    if ex.get("totals", {}).get("n_orders"):
        t = ex["totals"]
        print(f"\norders: {t['n_orders']} | fill rate {t['fill_rate']:.2f} | "
              f"traded ${t['traded_notional']:,.0f} | venue {ex['venue']}")
        by_status = ", ".join(f"{k} {v}" for k, v in sorted(t.get("by_status", {}).items()))
        print(f"  status: {by_status}")
        if t.get("reject_reasons"):
            for reason, n in sorted(t["reject_reasons"].items(), key=lambda kv: -kv[1])[:4]:
                print(f"  {n:4d}x {reason}")

    if report.get("revision"):
        r = report["revision"]
        print(f"\nround-2 revisions: flip rate {r['mean_flip_rate']:.3f}, "
              f"mean |delta| {r['mean_abs_delta_bps']:.1f}bps, "
              f"mean delta confidence {r['mean_delta_confidence']:+.3f}")
    if report.get("skipped"):
        reasons = {}
        for s in report["skipped"]:
            reasons[s["reason"].split(":")[0]] = reasons.get(s["reason"].split(":")[0], 0) + 1
        print(f"skipped cycles: {reasons}")
    print(f"\nrun log:  {report['log_path']}")


def cmd_backtest(args) -> None:
    universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    engine = make_engine(args, universe)

    def progress(i, day, record):
        c = record["consensus"]
        top = c["forecasts"][0] if c["forecasts"] else {}
        print(f"  [{i:3d}] {day} news={record['n_news']:2d} regime={c['risk_regime']:8} "
              f"top={top.get('ticker','-'):4} {top.get('direction','-'):5} "
              f"agree={c['mean_agreement']:.2f}", flush=True)

    resume_engine(args, engine)
    report = engine.run_backtest(args.start, args.end, step_days=args.step_days,
                                 max_cycles=args.max_cycles,
                                 on_cycle=None if args.quiet else progress)
    print_report(report)


def cmd_live(args) -> None:
    universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    args.feed = args.feed if args.feed != "archive" else "yahoo"
    engine = make_engine(args, universe)
    prices = PriceStore()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prices.ensure(universe + [engine.config.index_ticker], "2024-01-01", today)
    engine.prices = prices

    def progress(i, day, record):
        c = record["consensus"]
        print(f"  [cycle {i}] {record['timestamp']} news={record['n_news']} "
              f"regime={c['risk_regime']} agreement={c['mean_agreement']:.2f}", flush=True)
        for f in c["forecasts"][:5]:
            print(f"      {f['ticker']:5} {f['direction']:5} {f['expected_return_bps']:+8.1f}bps "
                  f"conf={f['confidence']:.2f} agree={f['agreement']:.2f}")

    resume_engine(args, engine)
    report = engine.run_live(interval_minutes=args.interval, cycles=args.cycles,
                             on_cycle=None if args.quiet else progress)
    print_report(report)


def cmd_ablate(args) -> None:
    """Same window, same news, different system configuration."""
    universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    rag_tool = RAGNewsTool()
    # Each variant turns off exactly one thing, so the column it moves is the thing.
    variants = {
        "full_panel":       dict(panel=dict(personas=DEFAULT_PANEL, rounds=2, use_rag=True)),
        "no_rag":           dict(panel=dict(personas=DEFAULT_PANEL, rounds=2, use_rag=False)),
        "no_communication": dict(panel=dict(personas=DEFAULT_PANEL, rounds=1, use_rag=True)),
        "single_agent":     dict(panel=dict(personas=["macro_econ"], rounds=1, use_rag=True)),
        "no_briefs":        dict(panel=dict(personas=DEFAULT_PANEL, rounds=2, use_rag=True),
                                 engine={"use_briefs": False}),
        "no_memory":        dict(panel=dict(personas=DEFAULT_PANEL, rounds=2, use_rag=True),
                                 engine={"reflect": False}),
    }
    chosen = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else list(variants)
    if args.mode != "llm":
        print("[warn] --mode is not 'llm': the mock and hardcoded agents ignore retrieved context,\n"
              "       briefs and reflections, so no_rag / no_briefs / no_memory will come out\n"
              "       identical to full_panel. Use --mode llm for those three to mean anything.")

    results = {}
    for name in chosen:
        if name not in variants:
            raise SystemExit(f"Unknown variant {name}. Available: {sorted(variants)}")
        print(f"\n--- ablation: {name} ---", flush=True)
        variant = variants[name]
        panel = make_panel(args, rag_tool, **variant["panel"])
        # Benchmarks are identical across variants; only run them once.
        engine = make_engine(args, universe, panel=panel, rag_tool=rag_tool,
                             run_id=f"ablate_{name}_{datetime.now():%Y%m%dT%H%M%S}",
                             benchmarks=None if name == chosen[0] else [])
        for key, value in (variant.get("engine") or {}).items():
            setattr(engine.config, key, value)
        report = engine.run_backtest(args.start, args.end, step_days=args.step_days,
                                     max_cycles=args.max_cycles)
        results[name] = report
        print_report(report)

    print("\n=== ablation summary (agents_consensus) ===")
    print(f"{'variant':20} {'hit':>7} {'wtd hit':>8} {'brier':>7} {'return':>9} {'sharpe':>8} "
          f"{'orders':>7}")
    rows = {}
    for name, rep in results.items():
        fm = rep["forecast_metrics"].get("agents_consensus", {})
        pm = rep["portfolio_metrics"].get("agents_consensus", {})
        orders = (rep.get("execution", {}).get("totals", {}) or {}).get("n_orders", 0)
        rows[name] = {"hit_rate": fm.get("hit_rate"), "brier": fm.get("brier_score"),
                      "cumulative_return": pm.get("cumulative_return"), "sharpe": pm.get("sharpe"),
                      "orders": orders,
                      "sim_vs_actual": rep.get("sim_vs_actual", {})}
        print(f"{name:20} {(fm.get('hit_rate') or 0):7.3f} "
              f"{(fm.get('confidence_weighted_hit_rate') or 0):8.3f} "
              f"{(fm.get('brier_score') or 0):7.3f} "
              f"{(pm.get('cumulative_return') or 0) * 100:8.2f}% {(pm.get('sharpe') or 0):8.2f} "
              f"{orders:7d}")

    out = Path(args.out or f"Agent persona/data/runs/ablation_{datetime.now():%Y%m%dT%H%M%S}.json")
    save_json(out, {"variants": rows, "reports": {k: v["run_id"] for k, v in results.items()}})
    print(f"\nwrote {out}")


def cmd_score(args) -> None:
    """Score a finished run once the market has moved (live runs need this)."""
    logs = sorted(Path(args.runs_dir).glob("run_*.jsonl")) if not args.log else [Path(args.log)]
    if not logs:
        raise SystemExit(f"No run logs found in {args.runs_dir}")
    log = logs[-1]

    report = rescore_run(log, horizon=args.horizon or "", index_ticker=args.index,
                         refresh=args.refresh)
    print(f"\n=== scored {report['run_id']} ({Path(report['run_log']).name}) ===")
    print(f"horizon {report['horizon']} on {report['bar_interval']} bars | "
          f"{report['cycles_scored']}/{report['cycles_in_log']} cycles scorable"
          + (f" ({report['cycles_not_yet_scorable']} still waiting on prices)"
             if report["cycles_not_yet_scorable"] else ""))

    fm = report["forecast_metrics"]
    if fm:
        print(f"\n{'model':28} {'n':>5} {'hit':>7} {'wtd hit':>8} {'brier':>7} {'MAE bps':>8}")
        for name, m in sorted(fm.items(), key=lambda kv: -(kv[1].get("hit_rate") or 0)):
            print(f"{name:28} {m['n_directional']:5d} {(m['hit_rate'] or 0):7.3f} "
                  f"{(m['confidence_weighted_hit_rate'] or 0):8.3f} "
                  f"{(m['brier_score'] or 0):7.3f} {(m['mean_abs_error_bps'] or 0):8.1f}")

    sva = report["sim_vs_actual"]
    print(f"\nsim vs actual: regime {sva['regime_accuracy']} on {sva['regime_calls']} calls | "
          f"direction {sva['directional_accuracy']} on {sva['directional_calls']} | "
          f"expected {sva['mean_expected_bps']:+.1f}bps vs actual {sva['mean_actual_bps']:+.1f}bps "
          f"(bias {sva['magnitude_bias_bps']:+.1f})")

    out = Path(args.out) if args.out else log.with_name(log.stem + "_scored.json")
    save_json(out, report)
    print(f"\nwrote {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="News-driven multi-agent simulation")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--universe", default=DEFAULT_UNIVERSE)
        sp.add_argument("--mode", choices=["llm", "mock", "hardcoded"], default="mock")
        sp.add_argument("--model", default="minimax/minimax-m3:free")
        sp.add_argument("--agents", default=",".join(DEFAULT_PANEL))
        sp.add_argument("--rounds", type=int, default=2)
        sp.add_argument("--horizon", default="1d")
        sp.add_argument("--no-rag", action="store_true")
        sp.add_argument("--digest-mode", choices=["heuristic", "llm"], default="heuristic")
        sp.add_argument("--feed", default="archive",
                        choices=["archive", "yahoo", "yahoo_rss", "finnhub", "fixture", "live"])
        sp.add_argument("--fixture", default="")
        sp.add_argument("--benchmarks", default="default",
                        help=f"'default' ({','.join(DEFAULT_BENCHMARKS)}), 'none', or a comma list")
        sp.add_argument("--cash", type=float, default=100000.0)
        sp.add_argument("--cost-bps", type=float, default=1.0)
        sp.add_argument("--max-gross", type=float, default=1.0)
        sp.add_argument("--long-only", action="store_true")
        sp.add_argument("--lookback-hours", type=float, default=24.0)
        sp.add_argument("--agent-cash", type=float, default=100000.0,
                        help="starting cash in each agent's own book")
        sp.add_argument("--slippage-bps", type=float, default=2.0)
        sp.add_argument("--max-order-pct", type=float, default=0.35,
                        help="largest single order as a share of that agent's equity")
        sp.add_argument("--cooldown-cycles", type=int, default=0,
                        help="cycles an agent must wait before trading the same ticker again")
        sp.add_argument("--no-agent-books", action="store_true",
                        help="skip order execution; score forecasts only")
        sp.add_argument("--agents-may-short", action="store_true",
                        help="let agents sell shares they do not hold (off: buy/sell only)")
        sp.add_argument("--town-crier", choices=["auto", "llm", "heuristic"], default="auto",
                        help="who writes the segment summary and the retrieval questions")
        sp.add_argument("--no-briefs", action="store_true",
                        help="ablation: hand agents the raw digest and let each retrieve for itself")
        sp.add_argument("--no-live-index", action="store_true",
                        help="do not index this run's news for later retrieval")
        sp.add_argument("--context-top-k", type=int, default=8,
                        help="documents retrieved for the brief's historical context")
        sp.add_argument("--reflection-mode", choices=["auto", "llm", "heuristic"], default="auto",
                        help="who writes the end-of-day reflection")
        sp.add_argument("--no-reflections", action="store_true",
                        help="ablation: agents keep no memory of their own past days")
        sp.add_argument("--reflections-top-k", type=int, default=3,
                        help="past lessons recalled into each brief")
        sp.add_argument("--reflect-every", type=int, default=1,
                        help="cycles between reflections (1 = every cycle, 2 = twice a day at 15m)")
        sp.add_argument("--max-position-pct", type=float, default=0.5,
                        help="largest resulting position in one name, as a share of equity")
        sp.add_argument("--memory-scope", choices=["run", "global"], default="run",
                        help="'run' keeps memory inside this run; 'global' remembers earlier runs")
        sp.add_argument("--quiet", action="store_true")
        sp.add_argument("--resume", default="",
                        help="continue a previous run id: restores books, cycle count, "
                             "trades and reflections (a 15-minute loop survives a restart)")

    bt = sub.add_parser("backtest", help="replay historical sessions")
    common(bt)
    bt.add_argument("--start", default="2016-01-04")
    bt.add_argument("--end", default="2016-06-30")
    bt.add_argument("--step-days", type=int, default=1)
    bt.add_argument("--max-cycles", type=int, default=None)
    bt.set_defaults(func=cmd_backtest)

    lv = sub.add_parser("live", help="poll live news on an interval")
    common(lv)
    lv.add_argument("--interval", type=int, default=15, help="minutes between cycles")
    lv.add_argument("--cycles", type=int, default=1)
    lv.set_defaults(func=cmd_live)

    ab = sub.add_parser("ablate", help="compare system configurations on one window")
    common(ab)
    ab.add_argument("--start", default="2016-01-04")
    ab.add_argument("--end", default="2016-03-31")
    ab.add_argument("--step-days", type=int, default=1)
    ab.add_argument("--max-cycles", type=int, default=None)
    ab.add_argument("--variants", default="")
    ab.add_argument("--out", default="")
    ab.set_defaults(func=cmd_ablate)

    sc = sub.add_parser("score", help="score a finished run once prices exist (live runs need this)")
    sc.add_argument("--log", default="", help="run log path (default: newest in --runs-dir)")
    sc.add_argument("--runs-dir", default="Agent persona/data/runs")
    sc.add_argument("--horizon", default="", help="override the horizon recorded in the log")
    sc.add_argument("--index", default="SPY")
    sc.add_argument("--refresh", action="store_true", help="re-download prices")
    sc.add_argument("--out", default="")
    sc.set_defaults(func=cmd_score, mode="mock")
    return p


def main() -> None:
    load_env()
    args = build_parser().parse_args()
    if args.mode == "llm" and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set (see .env.example). Use --mode mock to run offline.")
    args.func(args)


if __name__ == "__main__":
    main()
