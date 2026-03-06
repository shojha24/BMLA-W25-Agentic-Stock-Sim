from __future__ import annotations

import json
from typing import Any, Dict

from core.types import Digest, State, AgentOutput
from agents.base import BaseAgent
from llm.openrouter_client import OpenRouterClient


SYSTEM = "You are a trading agent. Output ONLY valid JSON. No markdown. No extra text."

DEVELOPER = """
Persona: Trend Follower.
Style: price-action and continuation. Likes strength, avoids weakness.
Focus:
- If returns fields exist in state (returns_1h / returns_1d / returns_1w), rely on them heavily.
- If returns are NOT provided, infer weak trends using sentiment/tickers in the digest and be conservative.
Rules:
- Use ONLY tickers in state.prices (do not invent tickers).
- Keep it concise: <= 8 signals, <= 6 trade_ideas.
- trade_ideas are opinions only (not executable orders).
Output a single JSON object with keys:
agent_name, persona, timestamp, decision,
market_view {risk_regime, confidence, summary, key_drivers},
signals [{news_id, headline, macro_tags, sentiment, confidence, tickers_mentioned, interpretation}],
trade_ideas [{ticker, bias, rationale, news_refs, suggested_position_pct_equity}],
checks {equity_estimate_usd, universe_prices_keys, digest_items}.
JSON only.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError("Model did not return JSON.")
    return json.loads(text[i : j + 1])


class TrendFollowerLLMAgent(BaseAgent):
    name = "trend_follower_llm_v1"
    persona = "Trend Follower (continuation; uses returns if provided; universe-limited)"

    def __init__(self, client: OpenRouterClient, model: str = "xiaomi/mimo-v2-flash:free"):
        self.client = client
        self.model = model

    def run(self, digest: Digest, state: State) -> AgentOutput:
        payload = {
            "timestamp": digest.get("timestamp", ""),
            "news_digest": digest.get("news_digest", []),
            "state": state,
            "hints": {
                "universe": sorted(list((state.get("prices") or {}).keys())),
                "available_return_fields": [k for k in ["returns_1h", "returns_1d", "returns_1w"] if k in (state or {})],
                "note": "Prefer uptrends, avoid downtrends. If no returns, stay modest and use ETF proxies first.",
            },
        }

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "developer", "content": DEVELOPER},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        content = self.client.chat(model=self.model, messages=messages, temperature=0.1)
        out = _extract_json(content)

        out.setdefault("agent_name", self.name)
        out.setdefault("persona", self.persona)
        out.setdefault("timestamp", payload["timestamp"])
        return out