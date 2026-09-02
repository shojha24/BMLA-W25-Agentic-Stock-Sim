"""The 15-Minute Brief.

One object per agent per cycle, assembled from four sources:

    Town Crier      news summary, stocks discussed, retrieved historical context
    Agent Assets    your balance (private - only your own book)
    Agent Actions   your last trades, and what your peers did (public)
    Reflections     your performance reflections (Phase 3; the slot is here and
                    stays empty until the reflection pipeline exists)

Before this, agents were handed a raw digest and their own price map. The brief
is what makes an agent aware of itself: what it holds, what it just did, and how
that worked out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from agents.town_crier import SegmentBrief
from sim.actions_db import ActionsDB
from sim.assets_db import AssetsDB

TRADE_FIELDS = ("day", "timestamp", "ticker", "side", "filled_qty", "price", "status", "reason")


@dataclass
class BriefConfig:
    max_last_trades: int = 5
    max_peer_actions: int = 6
    max_context_docs: int = 8
    include_peer_actions: bool = True
    include_rejected_trades: bool = True     # an agent should learn what got refused


class BriefAssembler:
    def __init__(self, actions_db: Optional[ActionsDB] = None,
                 assets_db: Optional[AssetsDB] = None,
                 config: Optional[BriefConfig] = None):
        self.actions_db = actions_db
        self.assets_db = assets_db
        self.config = config or BriefConfig()

    @staticmethod
    def _trim(rows: Sequence[Dict[str, Any]], fields=TRADE_FIELDS,
              with_agent: bool = False) -> List[Dict[str, Any]]:
        keys = (("agent_id",) + fields) if with_agent else fields
        return [{k: r.get(k) for k in keys} for r in rows]

    def build(
        self,
        agent_name: str,
        segment: SegmentBrief,
        state: Dict[str, Any],
        *,
        run_id: str = "",
        cycle: int = 0,
        historical_context: str = "",
        historical_docs: Optional[Sequence[Dict[str, Any]]] = None,
        reflections: Optional[Sequence[Dict[str, Any]]] = None,
        order_instructions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config

        last_trades: List[Dict[str, Any]] = []
        peer_actions: List[Dict[str, Any]] = []
        if self.actions_db and run_id:
            last_trades = self._trim(self.actions_db.last_trades(
                run_id, agent_name, limit=cfg.max_last_trades,
                filled_only=not cfg.include_rejected_trades))
            if cfg.include_peer_actions:
                peer_actions = self._trim(
                    self.actions_db.peer_actions(run_id, agent_name, limit=cfg.max_peer_actions),
                    with_agent=True)

        positions = state.get("positions", {}) or {}
        prices = state.get("prices", {}) or {}
        holdings = {
            ticker: {
                "qty": pos.get("qty", 0.0),
                "avg_price": pos.get("avg_price", 0.0),
                "market_value": round(pos.get("qty", 0.0) * prices.get(ticker, 0.0), 2),
                "unrealized_pnl": round(
                    (prices.get(ticker, 0.0) - pos.get("avg_price", 0.0)) * pos.get("qty", 0.0), 2),
            }
            for ticker, pos in positions.items() if pos.get("qty")
        }
        cash = float(state.get("cash_usd", 0.0) or 0.0)
        equity = round(cash + sum(h["market_value"] for h in holdings.values()), 2)

        # Do the arithmetic for the model rather than hoping it does it right: the
        # most common invalid order is selling shares the agent does not own.
        instructions = dict(order_instructions or {})
        max_pct = float(instructions.get("max_order_pct_equity", 1.0) or 1.0)
        budget = min(cash, equity * max_pct)
        instructions["you_can_sell_at_most"] = {
            ticker: int(h["qty"]) for ticker, h in holdings.items() if h["qty"] > 0
        }
        instructions["you_can_buy_at_most"] = {
            ticker: int(budget // price)
            for ticker, price in prices.items() if price > 0
        }

        return {
            "agent_id": agent_name,
            "timestamp": segment.timestamp,
            "cycle": cycle,
            "news_summary": segment.summary,
            "stocks_discussed": segment.stocks,
            "news_digest": segment.digest.get("news_digest", []),
            "your_balance": {
                "cash_usd": round(cash, 2),
                "equity_usd": equity,
                "positions": holdings,
                "prices": prices,
            },
            "your_last_trades": last_trades,
            "peer_recent_actions": peer_actions,
            "historical_context": {
                "summary": historical_context,
                "documents": list(historical_docs or [])[: cfg.max_context_docs],
                "questions_asked": segment.rag_questions.get("news", []),
            },
            "your_reflections": list(reflections or []),
            "order_instructions": instructions,
        }
