from __future__ import annotations

import argparse
import json
import os
import sys

# --- Make imports work when running this file directly (VSCode Run) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Agent persona/src
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# ---------------------------------------------------------------------

from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(BASE_DIR), ".env")
print("DEBUG ENV_PATH =", ENV_PATH)
print("DEBUG ENV exists =", os.path.exists(ENV_PATH))

ok = load_dotenv(ENV_PATH)
print("DEBUG load_dotenv returned =", ok)
print("DEBUG OPENROUTER_API_KEY loaded? =", bool(os.getenv("OPENROUTER_API_KEY")))
from core.io import load_json


def resolve_path(p: str) -> str:
    """Allow paths relative to cwd OR relative to project root."""
    if os.path.exists(p):
        return p
    alt = os.path.join(os.path.dirname(BASE_DIR), p)  # project_root/p
    if os.path.exists(alt):
        return alt
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--digest", default="data/digest.json", help="Path to digest JSON")
    parser.add_argument("--state", default="data/state.json", help="Path to state JSON")

    parser.add_argument("--mode", choices=["llm", "hardcoded"], default="llm",
                        help="Use OpenRouter LLM agent or hardcoded baseline.")
    parser.add_argument("--model", default="xiaomi/mimo-v2-flash:free",
                        help="OpenRouter model id (e.g. xiaomi/mimo-v2-flash:free)")

    args = parser.parse_args()

    digest_path = resolve_path(args.digest)
    state_path = resolve_path(args.state)

    digest = load_json(digest_path)
    state = load_json(state_path)

    if args.mode == "hardcoded":
        from agents.macro_econ import MacroEconomistAgent
        agent = MacroEconomistAgent()
        out = agent.run(digest=digest, state=state)
    else:
        from llm.openrouter_client import OpenRouterClient
        from agents.macro_econ import MacroEconomistLLMAgent

        key = os.getenv("OPENROUTER_API_KEY")
        client = OpenRouterClient(api_key=key, site_url="http://localhost", app_title="Sentiment Alpha")
        agent = MacroEconomistLLMAgent(client=client, model=args.model)
        out = agent.run(digest=digest, state=state)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()