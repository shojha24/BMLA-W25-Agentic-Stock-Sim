from __future__ import annotations
import json
from typing import Any, Dict

from core.types import Digest, State, AgentOutput
from agents.base import BaseAgent
from llm.openrouter_client import OpenRouterClient


MACRO_SYSTEM = (
    "You are a Macro Economist trading agent. "
    "You must output ONLY valid JSON (no markdown, no extra text)."
)

MACRO_DEVELOPER = """
Persona: Macro Economist (rates/inflation/growth/FX/commodities). Prefer liquid ETFs & mega-caps. Avoid illiquid names.
Task: Given (1) a 15-minute news digest and (2) current portfolio state, output a SINGLE JSON object with:
- agent_name, persona, timestamp, decision
- market_view: risk_regime (RISK_ON/RISK_OFF/NEUTRAL), confidence (0..1), summary, key_drivers (list of strings referencing news_id)
- signals: list of objects summarizing each news item and interpretation (include news_id, macro_tags, sentiment, confidence)
- trade_ideas: list of structured opinions (ticker, bias OVERWEIGHT/UNDERWEIGHT/NEUTRAL, rationale, news_refs, suggested_position_pct_equity)
- checks: equity_estimate_usd, universe_prices_keys, digest_items

Hard constraints:
- Use only tickers that appear in state.prices (do not invent tickers).
- trade_ideas are opinions only (no need to compute share quantities).
- Keep it concise: <= 8 signals, <= 6 trade_ideas.
Output JSON ONLY.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    # fallback: find first JSON object
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError("Model did not return JSON.")
    return json.loads(text[i : j + 1])


class MacroEconomistLLMAgent(BaseAgent):
    name = "macro_econ_llm_v1"
    persona = "Macro Economist (rates/inflation/growth/FX/commodities; prefers liquid ETFs & mega-caps)"

    def __init__(self, client: OpenRouterClient, model: str = "xiaomi/mimo-v2-flash:free"):
        self.client = client
        self.model = model

    def run(self, digest: Digest, state: State) -> AgentOutput:
        # Provide the LLM only what it needs
        payload = {
            "timestamp": digest.get("timestamp", ""),
            "news_digest": digest.get("news_digest", []),
            "state": state,
        }

        messages = [
            {"role": "system", "content": MACRO_SYSTEM},
            {"role": "developer", "content": MACRO_DEVELOPER},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        content = self.client.chat(model=self.model, messages=messages, temperature=0.0)
        out = _extract_json(content)

        # Minimal sanity checks (you can expand later)
        out.setdefault("agent_name", self.name)
        out.setdefault("persona", self.persona)
        return out  # AgentOutput dict