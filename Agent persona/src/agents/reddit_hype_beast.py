from __future__ import annotations

import json
from typing import Any, Dict

from core.types import Digest, State, AgentOutput
from agents.base import BaseAgent
from llm.openrouter_client import OpenRouterClient


SYSTEM = "You are a trading agent. Output ONLY valid JSON. No markdown. No extra text."

DEVELOPER = """
Persona: Reddit Hype-Beast.
Style: narrative-driven, buzz-sensitive, momentum-chasing, high-beta bias.
Focus:
- React to hype, virality, crowd sentiment, and “what’s hot”.
- Prefer QQQ / NVDA / mega-cap tech when sentiment is bullish.
- If risk-off tone dominates, become cautious quickly (trim risk).
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


class RedditHypeBeastLLMAgent(BaseAgent):
    name = "reddit_hype_beast_llm_v1"
    persona = "Reddit Hype-Beast (buzz/memes/momentum; high-beta; universe-limited)"

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
                "note": "Even if sources are not literally Reddit yet, behave like a hype-driven trader using sentiment + headlines.",
            },
        }

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "developer", "content": DEVELOPER},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        content = self.client.chat(model=self.model, messages=messages, temperature=0.3)
        out = _extract_json(content)

        out.setdefault("agent_name", self.name)
        out.setdefault("persona", self.persona)
        out.setdefault("timestamp", payload["timestamp"])
        return out