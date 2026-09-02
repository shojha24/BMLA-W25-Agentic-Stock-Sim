"""The reflection pipeline: an agent reads its own day back.

Whiteboard: "Retrieve market info for the day, all trades made by agent ->
Agent Per. A Reflect. -> Reflection -> Store in Vector DB Reflections".

An agent reflects on the *previous* cycle, not the current one, because a day's
trades cannot be judged until the market has moved. The engine holds each
cycle's trades and forecasts as pending and settles them once the outcome is
known, so a reflection never contains information the agent could not have had.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.schema import extract_json

REFLECT_SYSTEM = "You are a trader reviewing your own day. Output ONLY a JSON object."

REFLECT_PROMPT = """
You are {persona}.

You are shown one trading day of your own activity: the orders you sent (including
any the venue refused), the forecasts you made, what the market actually did, and
what happened to your book. Write the note you would want to read the next time a
similar setup appears.

Return {{"lesson": string, "what_worked": [string, ...], "what_failed": [string, ...],
"tags": [UPPERCASE_TAG, ...], "tickers": [ticker, ...]}}

- "lesson": 1-3 sentences, specific and self-critical. Name tickers and numbers.
  A lesson that would not change a future decision is not worth storing - if the day
  was genuinely uninformative, say exactly that in one line.
- Judge the decision, not the outcome: a correct call that lost money and a lucky
  guess that made money are both worth flagging as such.
- "tags": the conditions this lesson applies to (e.g. CPI, EVENT_RISK, THIN_TAPE),
  because that is how you will find it again.
"""


@dataclass
class DayRecord:
    """One cycle of an agent's activity, held until its outcome is known."""
    agent_id: str
    persona: str
    day: str
    timestamp: str
    trades: List[Dict[str, Any]] = field(default_factory=list)
    forecasts: List[Dict[str, Any]] = field(default_factory=list)
    equity: float = 0.0
    positions: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_activity(self) -> bool:
        return bool(self.trades or self.positions)


def _trade_verdicts(trades: Sequence[Dict[str, Any]],
                    realized_bps: Dict[str, float]) -> List[Dict[str, Any]]:
    """Did each trade point the same way the market subsequently moved?"""
    out = []
    for t in trades:
        if t.get("status") == "REJECTED":
            out.append({"ticker": t["ticker"], "side": t["side"], "status": "REJECTED",
                        "reason": t.get("reason", ""), "verdict": "not filled"})
            continue
        moved = realized_bps.get(str(t["ticker"]).upper())
        verdict = "unknown"
        if moved is not None:
            right = (moved > 0) if t["side"] == "BUY" else (moved < 0)
            verdict = "right" if right else "wrong"
        out.append({"ticker": t["ticker"], "side": t["side"], "qty": t.get("filled_qty"),
                    "price": t.get("price"), "realized_bps": None if moved is None else round(moved, 1),
                    "verdict": verdict})
    return out


def _forecast_verdicts(forecasts: Sequence[Dict[str, Any]],
                       realized_bps: Dict[str, float]) -> List[Dict[str, Any]]:
    out = []
    for f in forecasts:
        moved = realized_bps.get(str(f.get("ticker", "")).upper())
        if moved is None or f.get("direction") == "FLAT":
            continue
        right = (moved > 0) == (f["direction"] == "UP")
        out.append({"ticker": f["ticker"], "direction": f["direction"],
                    "expected_bps": f.get("expected_return_bps"),
                    "realized_bps": round(moved, 1), "correct": right,
                    "confidence": f.get("confidence")})
    return out


class ReflectionAgent:
    name = "reflection"

    def __init__(self, client=None, model: str = "", use_llm: bool = True):
        self.client = client
        self.model = model
        self.use_llm = use_llm and client is not None

    def reflect(self, record: DayRecord, realized_bps: Dict[str, float],
                equity_now: float) -> Optional[Dict[str, Any]]:
        if not record.has_activity:
            return None                       # nothing happened; nothing to learn

        pnl = round(equity_now - record.equity, 2)
        trades = _trade_verdicts(record.trades, realized_bps)
        forecasts = _forecast_verdicts(record.forecasts, realized_bps)
        heuristic = self._heuristic(record, trades, forecasts, pnl)

        if not self.use_llm:
            return heuristic

        payload = {
            "day": record.day,
            "your_trades": trades,
            "your_forecasts": forecasts,
            "market_moves_bps": {k: round(v, 1) for k, v in realized_bps.items()},
            "pnl_usd": pnl,
            "equity_before": round(record.equity, 2),
            "equity_after": round(equity_now, 2),
            "open_positions": record.positions,
        }
        try:
            content = self.client.chat(model=self.model, messages=[
                {"role": "system", "content": REFLECT_SYSTEM},
                {"role": "developer", "content": REFLECT_PROMPT.format(persona=record.persona)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ], temperature=0.2)
            raw = extract_json(content)
        except Exception:
            return heuristic

        lesson = str(raw.get("lesson") or "").strip()
        if not lesson:
            return heuristic
        return {
            "lesson": lesson[:800],
            "what_worked": [str(x)[:200] for x in (raw.get("what_worked") or [])][:4],
            "what_failed": [str(x)[:200] for x in (raw.get("what_failed") or [])][:4],
            "tags": [str(t).upper()[:32] for t in (raw.get("tags") or [])][:6] or heuristic["tags"],
            "tickers": [str(t).upper()[:8] for t in (raw.get("tickers") or [])][:8]
                       or heuristic["tickers"],
            "pnl_usd": pnl,
            "n_trades": len([t for t in trades if t.get("verdict") != "not filled"]),
            "source": "llm",
        }

    @staticmethod
    def _heuristic(record: DayRecord, trades: List[Dict[str, Any]],
                   forecasts: List[Dict[str, Any]], pnl: float) -> Dict[str, Any]:
        filled = [t for t in trades if t.get("verdict") in ("right", "wrong")]
        right = [t for t in filled if t["verdict"] == "right"]
        rejected = [t for t in trades if t.get("verdict") == "not filled"]
        hits = [f for f in forecasts if f["correct"]]

        parts = [f"{record.day}: P&L ${pnl:+,.2f}."]
        if filled:
            parts.append(f"{len(right)}/{len(filled)} trades moved my way "
                         f"({', '.join(t['ticker'] for t in filled)}).")
        else:
            parts.append("No fills to judge.")
        if forecasts:
            parts.append(f"Forecasts {len(hits)}/{len(forecasts)} correct.")
        if rejected:
            parts.append(f"{len(rejected)} order(s) refused: "
                         f"{'; '.join(sorted({t.get('reason', '') for t in rejected}))}.")

        return {
            "lesson": " ".join(parts),
            "what_worked": [f"{t['ticker']} {t['side']}" for t in right][:4],
            "what_failed": [f"{t['ticker']} {t['side']}" for t in filled
                            if t["verdict"] == "wrong"][:4],
            "tags": ["PROFIT" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT_DAY"],
            "tickers": sorted({t["ticker"] for t in trades}),
            "pnl_usd": pnl,
            "n_trades": len(filled),
            "source": "heuristic",
        }
