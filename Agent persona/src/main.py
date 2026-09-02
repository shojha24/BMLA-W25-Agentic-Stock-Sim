"""Single-cycle inspector: run the panel once on a fixture digest and look at it.

Same pipeline as `simulate.py`, minus the market: Town Crier -> retrieval ->
15-Minute Brief -> roundtable -> consensus -> orders through the venue. Prices
and balances come from `data/state.json` rather than from market data, so this
runs anywhere and is the fastest way to see what one cycle produces.

For anything with a clock, a P&L or a benchmark, use simulate.py.

  python "Agent persona/src/main.py" --task roundtable --mode mock
  python "Agent persona/src/main.py" --task roundtable --mode llm
  python "Agent persona/src/main.py" --task rag-status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent           # .../Agent persona/src
PROJECT_DIR = BASE_DIR.parent                        # .../Agent persona
REPO_ROOT = PROJECT_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.baseline import SentimentBaselineAgent      # noqa: E402
from agents.persona import LLMPersonaAgent              # noqa: E402
from agents.personas import DEFAULT_PANEL, PERSONAS     # noqa: E402
from agents.town_crier import TownCrierAgent            # noqa: E402
from core.io import load_env, load_json, save_json      # noqa: E402
from data.digest_builder import HeuristicDigestBuilder  # noqa: E402
from data.news_feed import FixtureNewsFeed              # noqa: E402
from orchestration.consensus import build_consensus     # noqa: E402
from orchestration.roundtable import Roundtable         # noqa: E402
from sim.brief import BriefAssembler                    # noqa: E402
from sim.execution import ExecutionConfig, MarketFillVenue, summarize_fills  # noqa: E402
from sim.portfolio import Portfolio                     # noqa: E402
from tools.rag import RAGNewsTool                       # noqa: E402


def resolve_path(p: str) -> Path:
    """Accept paths relative to cwd, to 'Agent persona/', or to the repo root."""
    for candidate in (Path(p), PROJECT_DIR / p, REPO_ROOT / p):
        if candidate.exists():
            return candidate
    return Path(p)


def build_client(mode: str, horizon: str):
    if mode == "mock":
        from llm.mock_client import MockChatClient
        return MockChatClient(horizon=horizon)
    from llm.openrouter_client import OpenRouterClient
    return OpenRouterClient(api_key=os.getenv("OPENROUTER_API_KEY"),
                            site_url="http://localhost", app_title="Agentic Stock Sim")


def build_agents(args, rag_tool, client):
    if args.mode == "hardcoded":
        return [SentimentBaselineAgent(horizon=args.horizon)]

    import dataclasses
    agents = []
    for key in [k.strip() for k in args.agents.split(",") if k.strip()]:
        if key not in PERSONAS:
            raise SystemExit(f"Unknown persona '{key}'. Available: {sorted(PERSONAS)}")
        spec = PERSONAS[key]
        if args.horizon != spec.default_horizon:
            spec = dataclasses.replace(spec, default_horizon=args.horizon)
        agents.append(LLMPersonaAgent(spec=spec, client=client, model=args.model,
                                      rag_tool=rag_tool, use_rag=not args.no_rag,
                                      max_order_pct_equity=args.max_order_pct))
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one cycle of the agent panel")
    parser.add_argument("--task", choices=["roundtable", "panel", "single", "rag-status"],
                        default="roundtable",
                        help="roundtable = 2 rounds with peer review; panel = 1 independent round")
    parser.add_argument("--digest", default="data/digest.json")
    parser.add_argument("--state", default="data/state.json")
    parser.add_argument("--mode", choices=["llm", "mock", "hardcoded"], default="llm")
    parser.add_argument("--model", default="minimax/minimax-m3:free")
    parser.add_argument("--agents", default=",".join(DEFAULT_PANEL))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--horizon", default="1d")
    parser.add_argument("--no-rag", action="store_true", help="ablation: run without retrieval")
    parser.add_argument("--no-briefs", action="store_true",
                        help="ablation: hand agents the raw digest instead of a brief")
    parser.add_argument("--no-execute", action="store_true",
                        help="skip sending the agents' orders to the venue")
    parser.add_argument("--max-order-pct", type=float, default=0.35)
    parser.add_argument("--context-top-k", type=int, default=6)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--out", default="", help="write the full result JSON here")
    parser.add_argument("--quiet", action="store_true", help="print the consensus only")
    args = parser.parse_args()

    load_env()
    rag_tool = RAGNewsTool()
    if args.task == "rag-status":
        print(json.dumps(rag_tool.status(), indent=2))
        return

    if args.mode == "llm" and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Add 'OPENROUTER_API_KEY=sk-or-...' to .env "
            "(a bare key with no VAR= prefix is ignored), or run with --mode mock.")

    digest_path = resolve_path(args.digest)
    state = load_json(resolve_path(args.state))
    universe = list((state.get("prices") or {}).keys())

    client = build_client(args.mode, args.horizon) if args.mode != "hardcoded" else None
    agents = build_agents(args, rag_tool, client)
    panel = Roundtable(agents, rounds=1 if args.task in ("panel", "single") else args.rounds,
                       horizon=args.horizon, parallel=not args.no_parallel)
    if args.task == "single":
        panel = Roundtable(agents[:1], rounds=1, horizon=args.horizon)

    # Town Crier reads the segment and writes the retrieval questions.
    news = FixtureNewsFeed(digest_path).items
    crier = TownCrierAgent(HeuristicDigestBuilder(), client=client, model=args.model,
                           use_llm=args.mode == "llm", max_context_docs=args.context_top_k)
    segment = crier.summarize_segment(news, str(load_json(digest_path).get("timestamp", "")),
                                      universe)

    # Retrieval runs once for the desk, as in the simulator.
    docs, context_summary = [], ""
    if not args.no_rag:
        seen = set()
        for question in segment.rag_questions.get("news", [])[:3]:
            rows, _ = rag_tool.retrieve(query=question, top_k=args.context_top_k,
                                        stock_filter=universe)
            for row in rows:
                if row["doc_id"] not in seen:
                    seen.add(row["doc_id"])
                    docs.append(row)
        docs = docs[: args.context_top_k]
        context_summary = crier.summarize_context(docs, segment.summary)

    # Each agent gets its own book, seeded from state.json, and its own brief.
    books = {agent.name: Portfolio(float(state.get("cash_usd", 0.0)),
                                   state.get("positions"), cost_bps=1.0)
             for agent in agents}
    order_rules = {"max_order_pct_equity": args.max_order_pct, "shorting_allowed": False,
                   "whole_shares_only": True}
    brief_by_agent = None
    if not args.no_briefs:
        assembler = BriefAssembler()
        brief_by_agent = {
            agent.name: assembler.build(agent.name, segment, state,
                                        historical_context=context_summary,
                                        historical_docs=docs, reflections=[],
                                        order_instructions=order_rules)
            for agent in agents
        }

    table = panel.run(segment.digest, state,
                      state_by_agent={a.name: state for a in agents},
                      brief_by_agent=brief_by_agent)

    fills = []
    if not args.no_execute:
        venue = MarketFillVenue(ExecutionConfig(max_order_pct_equity=args.max_order_pct))
        venue.start_cycle()
        orders = {out["agent_name"]: out.get("orders", []) or []
                  for out in table["rounds"][-1]["outputs"]}
        fills = venue.execute(orders, state.get("prices", {}), books, segment.timestamp)

    result = {
        "segment": segment.to_dict(),
        "historical_context": {"summary": context_summary, "documents": docs},
        "consensus": table["consensus"],
        "consensus_round1": table.get("consensus_round1"),
        "agent_outputs": table["rounds"][-1]["outputs"],
        "revision": table.get("revision", {}),
        "fills": fills,
        "execution": summarize_fills(fills),
        "books": {name: book.snapshot(state.get("prices", {})) for name, book in books.items()},
        "errors": table.get("errors", []),
    }
    if args.task == "single":
        result["consensus"] = build_consensus(table["rounds"][-1]["outputs"],
                                              segment.timestamp, args.horizon)

    if args.out:
        save_json(resolve_path(args.out) if Path(args.out).is_absolute() else Path(args.out), result)
        print(f"wrote {args.out}")

    if args.quiet:
        print(json.dumps(result["consensus"], indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if result["errors"]:
        print(f"\n[warn] {len(result['errors'])} agent failure(s): "
              f"{json.dumps(result['errors'])[:400]}", file=sys.stderr)


if __name__ == "__main__":
    main()
