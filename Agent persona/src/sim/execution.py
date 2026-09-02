"""The market simulator: turns agent orders into fills against exogenous prices.

Prices come from the real market (historical or live) rather than forming from
the agents' own supply and demand. With a handful of agents there is no
liquidity to make a market, and the point of the simulation is to compare the
agents' book against what the market actually did - which only works if the
market is the market.

`ExecutionVenue` is an interface so a limit-order book with endogenous price
formation can replace `MarketFillVenue` later without touching the engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.types import Fill, Order
from sim.portfolio import Portfolio


@dataclass
class ExecutionConfig:
    cost_bps: float = 1.0            # commission / fees on traded notional
    slippage_bps: float = 2.0        # price concession, charged against the taker
    allow_short: bool = False        # whiteboard flow is buy/sell of held shares
    whole_shares: bool = True
    max_order_pct_equity: float = 0.35   # largest single order
    max_position_pct_equity: float = 0.5  # largest resulting position in one name
    cooldown_cycles: int = 0         # per agent+ticker; 0 disables
    min_notional: float = 1.0


class ExecutionVenue(ABC):
    name = "venue"

    @abstractmethod
    def execute(
        self,
        orders_by_agent: Dict[str, Sequence[Order]],
        prices: Dict[str, float],
        books: Dict[str, Portfolio],
        timestamp: str,
    ) -> List[Fill]:
        ...


class MarketFillVenue(ExecutionVenue):
    """Fill at the market price plus slippage, subject to cash and holdings.

    Every rejection carries a reason, because "the LLM asked for something
    impossible" is a result worth measuring, not an error to swallow.
    """

    name = "market_fill"

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self._cycle = 0
        self._last_trade_cycle: Dict[Tuple[str, str], int] = {}

    def start_cycle(self) -> None:
        self._cycle += 1

    # ---------------- helpers ----------------

    def _fill_price(self, price: float, side: str) -> float:
        slip = self.config.slippage_bps / 10000.0
        return price * (1.0 + slip) if side == "BUY" else price * (1.0 - slip)

    def _cooldown_blocked(self, agent_id: str, ticker: str) -> bool:
        if self.config.cooldown_cycles <= 0:
            return False
        last = self._last_trade_cycle.get((agent_id, ticker))
        return last is not None and (self._cycle - last) < self.config.cooldown_cycles

    @staticmethod
    def _reject(agent_id: str, timestamp: str, order: Order, reason: str,
                price: float = 0.0) -> Fill:
        return {
            "agent_id": agent_id, "timestamp": timestamp, "ticker": order["ticker"],
            "side": order["side"], "requested_qty": order["qty"], "filled_qty": 0.0,
            "price": round(price, 4), "notional": 0.0, "cost": 0.0,
            "status": "REJECTED", "reason": reason,
        }

    def _allowed_qty(self, order: Order, price: float, book: Portfolio) -> Tuple[float, str]:
        """Largest quantity this book can actually trade, and why it was cut."""
        cfg = self.config
        qty = order["qty"]
        reason = ""
        equity = book.equity({**{t: p["avg_price"] for t, p in book.positions.items()},
                              order["ticker"]: price})

        cap_qty = (equity * cfg.max_order_pct_equity) / price if price > 0 else 0.0
        if qty > cap_qty:
            qty, reason = cap_qty, f"capped at {cfg.max_order_pct_equity:.0%} of equity"

        # Concentration limit: repeated small buys must not add up to the whole book.
        if order["side"] == "BUY" and cfg.max_position_pct_equity > 0 and price > 0:
            held = book.positions.get(order["ticker"], {}).get("qty", 0.0)
            room = (equity * cfg.max_position_pct_equity) / price - held
            if qty > room:
                qty = max(room, 0.0)
                reason = f"position capped at {cfg.max_position_pct_equity:.0%} of equity"

        if order["side"] == "BUY":
            unit = price * (1.0 + cfg.slippage_bps / 10000.0) * (1.0 + cfg.cost_bps / 10000.0)
            affordable = book.cash / unit if unit > 0 else 0.0
            if qty > affordable:
                qty, reason = max(affordable, 0.0), "insufficient cash"
        else:
            held = book.positions.get(order["ticker"], {}).get("qty", 0.0)
            if not cfg.allow_short and qty > held:
                qty, reason = max(held, 0.0), "insufficient shares (shorting disabled)"

        if cfg.whole_shares:
            qty = float(int(qty))
        return qty, reason

    # ---------------- main ----------------

    def execute(
        self,
        orders_by_agent: Dict[str, Sequence[Order]],
        prices: Dict[str, float],
        books: Dict[str, Portfolio],
        timestamp: str,
    ) -> List[Fill]:
        cfg = self.config
        fills: List[Fill] = []

        for agent_id, orders in orders_by_agent.items():
            book = books.get(agent_id)
            if book is None:
                continue
            for raw in orders:
                order = _coerce(raw)
                if order is None:
                    continue
                ticker = order["ticker"]
                price = prices.get(ticker, 0.0)
                if price <= 0:
                    fills.append(self._reject(agent_id, timestamp, order, "no price for ticker"))
                    continue
                if self._cooldown_blocked(agent_id, ticker):
                    fills.append(self._reject(agent_id, timestamp, order,
                                              f"cooldown: traded within {cfg.cooldown_cycles} cycles", price))
                    continue
                if order["order_type"] == "LIMIT" and order.get("limit_price"):
                    marketable = (price <= order["limit_price"]) if order["side"] == "BUY" \
                        else (price >= order["limit_price"])
                    if not marketable:
                        fills.append(self._reject(agent_id, timestamp, order,
                                                  "limit not marketable at this price", price))
                        continue

                qty, cut_reason = self._allowed_qty(order, price, book)
                fill_price = self._fill_price(price, order["side"])
                if qty <= 0 or qty * fill_price < cfg.min_notional:
                    fills.append(self._reject(agent_id, timestamp, order,
                                              cut_reason or "quantity rounds to zero", price))
                    continue

                signed_qty = qty if order["side"] == "BUY" else -qty
                notional = signed_qty * fill_price
                cost = abs(notional) * cfg.cost_bps / 10000.0
                book.apply_fill(ticker, signed_qty, fill_price, cost)
                self._last_trade_cycle[(agent_id, ticker)] = self._cycle

                partial = qty < order["qty"] - 1e-9
                fills.append({
                    "agent_id": agent_id, "timestamp": timestamp, "ticker": ticker,
                    "side": order["side"], "requested_qty": order["qty"],
                    "filled_qty": round(qty, 4), "price": round(fill_price, 4),
                    "notional": round(-notional, 2), "cost": round(cost, 4),
                    "status": "PARTIAL" if partial else "FILLED",
                    "reason": cut_reason if partial else "",
                })
        return fills


def _coerce(raw: Any) -> Optional[Order]:
    """Fill in the optional fields of an order that skipped normalization."""
    if not isinstance(raw, dict) or not raw.get("ticker") or not raw.get("side"):
        return None
    order = dict(raw)
    order.setdefault("order_type", "MARKET")
    order.setdefault("limit_price", None)
    order.setdefault("qty", 0.0)
    order.setdefault("rationale", "")
    order.setdefault("news_refs", [])
    return order  # type: ignore[return-value]


def summarize_fills(fills: Sequence[Fill]) -> Dict[str, Any]:
    """How well did the agents follow the rules of the venue?"""
    if not fills:
        return {"n_orders": 0}
    by_status: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    for f in fills:
        by_status[f["status"]] = by_status.get(f["status"], 0) + 1
        if f["reason"]:
            reasons[f["reason"]] = reasons.get(f["reason"], 0) + 1
    return {
        "n_orders": len(fills),
        "by_status": by_status,
        "reject_reasons": reasons,
        "traded_notional": round(sum(abs(f["notional"]) for f in fills), 2),
        "fill_rate": round(sum(1 for f in fills if f["status"] != "REJECTED") / len(fills), 4),
    }
