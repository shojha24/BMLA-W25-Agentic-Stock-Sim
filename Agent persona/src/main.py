"""CLI entry point for the agent panel.

  python "Agent persona/src/main.py" --task roundtable --mode mock
  python "Agent persona/src/main.py" --task roundtable --mode llm --model minimax/minimax-m3:free
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
from core.io import load_env, load_json, save_json      # noqa: E402
from orchestration.consensus import build_consensus     # noqa: E402
from orchestration.roundtable import Roundtable         # noqa: E402
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
    return OpenRouterClient(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        site_url="http://localhost",
        app_title="Agentic Stock Sim",
    )


def build_agents(args, rag_tool):
    if args.mode == "hardcoded":
        return [SentimentBaselineAgent(horizon=args.horizon)]

    client = build_client(args.mode, args.horizon)
    keys = [k.strip() for k in args.agents.split(",") if k.strip()]
    unknown = [k for k in keys if k not in PERSONAS]
    if unknown:
        raise SystemExit(f"Unknown persona(s): {unknown}. Available: {sorted(PERSONAS)}")

    agents = []
    for key in keys:
        spec = PERSONAS[key]
        if args.horizon != spec.default_horizon:
            import dataclasses
            spec = dataclasses.replace(spec, default_horizon=args.horizon)
        agents.append(
            LLMPersonaAgent(
                spec=spec,
                client=client,
                model=args.model,
                rag_tool=rag_tool,
                use_rag=not args.no_rag,
            )
        )
    if args.with_baseline:
        agents.append(SentimentBaselineAgent(horizon=args.horizon))
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent news-driven forecast panel")
    parser.add_argument("--task", choices=["roundtable", "panel", "single", "rag-status"],
                        default="roundtable",
                        help="roundtable = 2 rounds with peer review; panel = 1 independent round")
    parser.add_argument("--digest", default="data/digest.json")
    parser.add_argument("--state", default="data/state.json")
    parser.add_argument("--mode", choices=["llm", "mock", "hardcoded"], default="llm",
                        help="llm = OpenRouter; mock = deterministic offline stub; hardcoded = rule-based baseline")
    parser.add_argument("--model", default="minimax/minimax-m3:free", help="OpenRouter model id")
    parser.add_argument("--agents", default=",".join(DEFAULT_PANEL),
                        help=f"comma-separated personas from {sorted(PERSONAS)}")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--horizon", default="1d")
    parser.add_argument("--no-rag", action="store_true", help="ablation: run without retrieval")
    parser.add_argument("--with-baseline", action="store_true",
                        help="add the rule-based agent to the panel as a control")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--out", default="", help="write the full result JSON here")
    parser.add_argument("--quiet", action="store_true", help="print consensus only")
    args = parser.parse_args()

    load_env()

    rag_tool = RAGNewsTool()
    if args.task == "rag-status":
        print(json.dumps(rag_tool.status(), indent=2))
        return

    digest = load_json(resolve_path(args.digest))
    state = load_json(resolve_path(args.state))

    if args.mode == "llm" and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Add 'OPENROUTER_API_KEY=sk-or-...' to .env "
            "(a bare key with no VAR= prefix is ignored), or run with --mode mock."
        )

    agents = build_agents(args, rag_tool)

    if args.task == "single":
        out = agents[0].run(digest, state)
        result = {"agent": out, "consensus": build_consensus([out], out.get("timestamp", ""), args.horizon)}
    else:
        rounds = 1 if args.task == "panel" else max(1, args.rounds)
        table = Roundtable(agents, rounds=rounds, horizon=args.horizon, parallel=not args.no_parallel)
        result = table.run(digest, state)

    if args.out:
        save_json(resolve_path(args.out) if Path(args.out).is_absolute() else Path(args.out), result)
        print(f"wrote {args.out}")

    if args.quiet:
        print(json.dumps(result.get("consensus", {}), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("errors"):
        print(f"\n[warn] {len(result['errors'])} agent failure(s): "
              f"{json.dumps(result['errors'])[:400]}", file=sys.stderr)


if __name__ == "__main__":
    main()
