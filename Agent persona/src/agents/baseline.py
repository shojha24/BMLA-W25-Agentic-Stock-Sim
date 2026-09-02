"""Rule-based agents: no LLM, no network.

`SentimentBaselineAgent` restores the `--mode hardcoded` path (which imported a
class that did not exist) and doubles as the naive benchmark: read the digest
sentiment, project it onto the universe, done.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from core.types import AgentOutput, Digest, Forecast, State
from core.utils import estimate_equity

_SENTIMENT_SIGN = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}

# Rough one-day beta of each instrument to a broad risk signal.
_RISK_BETA = {
    "SPY": 1.0, "QQQ": 1.25, "XLE": 0.9,
    "TLT": -0.6, "GLD": -0.3, "UUP": -0.4,
}


class SentimentBaselineAgent(BaseAgent):
    name = "sentiment_baseline_v1"
    persona = "Rule-based sentiment baseline (no LLM)"

    def __init__(self, scale_bps: float = 60.0, horizon: str = "1d"):
        self.scale_bps = scale_bps
        self.horizon = horizon

    def run(
        self,
        digest: Digest,
        state: State,
        peer_context: Optional[List[Dict[str, Any]]] = None,
        prior_output: Optional[AgentOutput] = None,
    ) -> AgentOutput:
        items = digest.get("news_digest", []) or []
        prices = state.get("prices", {}) or {}

        weighted = 0.0
        total_w = 0.0
        for item in items:
            sign = _SENTIMENT_SIGN.get(str(item.get("sentiment", "")).upper(), 0.0)
            conf = float(item.get("confidence", 0.5) or 0.5)
            weighted += sign * conf
            total_w += conf
        tone = weighted / total_w if total_w else 0.0

        forecasts: List[Forecast] = []
        for ticker in prices:
            beta = _RISK_BETA.get(ticker.upper(), 0.8)
            bps = round(tone * beta * self.scale_bps, 2)
            direction = "FLAT" if abs(bps) <= 10 else ("UP" if bps > 0 else "DOWN")
            forecasts.append({
                "ticker": ticker,
                "direction": direction,  # type: ignore[typeddict-item]
                "expected_return_bps": bps,
                "horizon": self.horizon,
                "confidence": round(min(0.75, abs(tone) * 0.8 + 0.15), 3),
                "rationale": f"digest tone {tone:+.2f} x risk beta {beta:+.2f}",
                "news_refs": [str(i.get("news_id", "")) for i in items][:4],
            })

        regime = "RISK_ON" if tone > 0.2 else ("RISK_OFF" if tone < -0.2 else "NEUTRAL")
        return {
            "agent_name": self.name,
            "persona": self.persona,
            "timestamp": str(digest.get("timestamp", "")),
            "decision": f"Baseline tone {tone:+.2f} -> {regime}",
            "market_view": {
                "risk_regime": regime,
                "confidence": round(min(0.8, abs(tone) + 0.2), 3),
                "summary": f"Sentiment-weighted read of {len(items)} digest items.",
                "key_drivers": [str(i.get("news_id", "")) for i in items][:5],
            },
            "signals": [
                {
                    "news_id": i.get("news_id", ""),
                    "macro_tags": i.get("macro_tags", []),
                    "sentiment": i.get("sentiment", "NEUTRAL"),
                    "confidence": float(i.get("confidence", 0.5) or 0.5),
                }
                for i in items[:8]
            ],
            "forecasts": forecasts,
            "trade_ideas": [],
            "checks": {
                "equity_estimate_usd": round(estimate_equity(state), 2),
                "universe_prices_keys": len(prices),
                "digest_items": len(items),
                "round": 1,
            },
        }
