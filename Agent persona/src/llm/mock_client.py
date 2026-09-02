"""Deterministic offline stand-in for a real LLM.

Lets the full pipeline (roundtable, consensus, evaluation) run and be unit
tested with no API key and no network. It is a stub, not a model: it reads the
digest sentiment and applies a fixed per-persona tilt.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from core.schema import extract_json

_SENTIMENT = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}

# How each persona reacts to the same news: trend-follow, fade, or take it straight.
_PERSONA_TILT = {
    "macro_econ": 1.0,
    "quant_momentum": 1.4,
    "contrarian_value": -0.8,
}
# Rough one-day sensitivity of each instrument to a broad risk signal, so the
# stub does not push gold, duration and equities in the same direction.
_RISK_BETA = {
    "SPY": 1.0, "QQQ": 1.25, "XLE": 0.9, "IWM": 1.2, "DIA": 0.95,
    "TLT": -0.6, "GLD": -0.3, "UUP": -0.4,
}

_PERSONA_MATCH = {
    "macro economist": "macro_econ",
    "quant momentum": "quant_momentum",
    "contrarian value": "contrarian_value",
}


def _jitter(seed: str) -> float:
    """Stable pseudo-random in [-0.5, 0.5] so mock runs are reproducible."""
    h = hashlib.md5(seed.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF - 0.5


def _mock_orders(forecasts, prices, positions, cash) -> List[Dict[str, Any]]:
    """Turn the stub's forecasts into orders it can actually pay for."""
    orders: List[Dict[str, Any]] = []
    budget = cash * 0.5                     # never spend the whole book in one cycle
    for f in sorted(forecasts, key=lambda x: -abs(x["expected_return_bps"])):
        ticker = f["ticker"]
        price = float(prices.get(ticker, 0.0) or 0.0)
        if price <= 0 or f["direction"] == "FLAT":
            continue
        held = float((positions.get(ticker) or {}).get("qty", 0.0))
        if f["direction"] == "UP":
            qty = int(min(budget, cash * 0.2) // price)
            if qty > 0:
                orders.append({"side": "BUY", "ticker": ticker, "qty": qty,
                               "rationale": f["rationale"], "news_refs": f["news_refs"]})
                budget -= qty * price
        elif held > 0:
            orders.append({"side": "SELL", "ticker": ticker, "qty": int(held),
                           "rationale": f["rationale"], "news_refs": f["news_refs"]})
    return orders[:4]


class MockChatClient:
    def __init__(self, horizon: str = "1d"):
        self.horizon = horizon
        self.calls: List[Dict[str, Any]] = []

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        self.calls.append({"model": model, "messages": messages})
        joined = " ".join(m.get("content", "") for m in messages).lower()

        persona_key = "macro_econ"
        for needle, key in _PERSONA_MATCH.items():
            if needle in joined:
                persona_key = key
                break

        payload: Dict[str, Any] = {}
        for m in reversed(messages):
            if m.get("role") == "user":
                try:
                    payload = extract_json(m.get("content", ""))
                except Exception:
                    payload = {}
                break

        return json.dumps(self._respond(persona_key, payload))

    def _respond(self, persona_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Agents receive either a raw digest or a 15-Minute Brief that nests it.
        brief = payload.get("brief") or {}
        digest = payload.get("news_digest") or brief.get("news_digest") or []
        state = payload.get("state") or {}
        if not state and brief.get("your_balance"):
            balance = brief["your_balance"]
            state = {"cash_usd": balance.get("cash_usd", 0.0),
                     "positions": balance.get("positions", {}),
                     "prices": balance.get("prices", {})}
        timestamp = payload.get("timestamp") or brief.get("timestamp") or ""
        universe = list((state.get("prices") or {}).keys())

        weight = sum(_SENTIMENT.get(str(i.get("sentiment", "")).upper(), 0.0)
                     * float(i.get("confidence", 0.5) or 0.5) for i in digest)
        tone = weight / max(len(digest), 1)
        tilt = _PERSONA_TILT.get(persona_key, 1.0)
        view = tone * tilt

        # Second round: drift a third of the way toward the peers' mean view.
        peers = payload.get("peer_forecasts") or []
        if peers:
            peer_bps = [float(p.get("expected_return_bps", 0.0)) for p in peers]
            if peer_bps:
                view = view + (sum(peer_bps) / len(peer_bps) / 100.0 - view) / 3.0

        refs = [str(i.get("news_id", "")) for i in digest][:4]
        forecasts = []
        for ticker in universe[:6]:
            beta = _RISK_BETA.get(ticker.upper(), 0.8)
            bps = round(view * beta * 60.0 + _jitter(f"{persona_key}|{ticker}|{timestamp}") * 25.0, 2)
            direction = "FLAT" if abs(bps) < 12 else ("UP" if bps > 0 else "DOWN")
            forecasts.append({
                "ticker": ticker,
                "direction": direction,
                "expected_return_bps": bps,
                "horizon": self.horizon,
                "confidence": round(min(0.9, 0.35 + abs(view) * 0.4), 3),
                "rationale": f"[mock:{persona_key}] tone={tone:+.2f} tilt={tilt:+.1f} beta={beta:+.2f}",
                "news_refs": refs,
            })

        prices = state.get("prices") or {}
        positions = state.get("positions") or {}
        cash = float(state.get("cash_usd", 0.0) or 0.0)
        actions = _mock_orders(forecasts, prices, positions, cash)

        regime = "RISK_ON" if view > 0.15 else ("RISK_OFF" if view < -0.15 else "NEUTRAL")
        return {
            "timestamp": timestamp,
            "decision": f"[mock:{persona_key}] net view {view:+.2f}",
            "market_view": {
                "risk_regime": regime,
                "confidence": round(min(0.9, 0.3 + abs(view) * 0.5), 3),
                "summary": f"Mock {persona_key} read of {len(digest)} digest items.",
                "key_drivers": refs,
            },
            "signals": [
                {
                    "news_id": i.get("news_id", ""),
                    "macro_tags": i.get("macro_tags", []),
                    "sentiment": i.get("sentiment", "NEUTRAL"),
                    "confidence": float(i.get("confidence", 0.5) or 0.5),
                    "interpretation": f"mock reading of {str(i.get('headline',''))[:60]}",
                }
                for i in digest[:8]
            ],
            "forecasts": forecasts,
            "actions": actions,
            "trade_ideas": [
                {
                    "ticker": f["ticker"],
                    "bias": {"UP": "OVERWEIGHT", "DOWN": "UNDERWEIGHT", "FLAT": "NEUTRAL"}[f["direction"]],
                    "rationale": f["rationale"],
                    "news_refs": refs,
                    "suggested_position_pct_equity": round(abs(f["expected_return_bps"]) / 2000.0, 4),
                }
                for f in forecasts[:4]
            ],
            "checks": {"digest_items": len(digest), "universe_prices_keys": len(universe)},
        }
