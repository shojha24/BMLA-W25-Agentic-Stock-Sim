"""Position sizing and a mark-to-market portfolio.

Weights come from the consensus: a direction the panel agrees on, scaled by how
confident it is, capped per name and in aggregate. Trades cross a spread; the
cost is charged on traded notional so that a strategy which churns every cycle
pays for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

DIRECTION_SIGN = {"UP": 1.0, "DOWN": -1.0, "FLAT": 0.0}


@dataclass
class Trade:
    ticker: str
    qty: float             # signed: positive = buy
    price: float
    notional: float
    cost: float


def target_weights(
    forecasts: Sequence[Dict[str, Any]],
    max_gross: float = 1.0,
    max_name: float = 0.34,
    min_confidence: float = 0.10,
    allow_short: bool = True,
) -> Dict[str, float]:
    """Consensus -> signed portfolio weights.

    Uses direction x confidence rather than expected_return_bps: LLM magnitude
    estimates are far noisier than their directional calls, so letting them size
    positions imports that noise into the P&L.
    """
    raw: Dict[str, float] = {}
    for f in forecasts:
        sign = DIRECTION_SIGN.get(str(f.get("direction", "FLAT")).upper(), 0.0)
        conf = float(f.get("confidence", 0.0) or 0.0)
        if sign == 0.0 or conf < min_confidence:
            continue
        if sign < 0 and not allow_short:
            continue
        w = sign * conf
        raw[str(f.get("ticker", "")).upper()] = max(-max_name, min(max_name, w))

    gross = sum(abs(w) for w in raw.values())
    if gross > max_gross and gross > 0:
        scale = max_gross / gross
        raw = {t: w * scale for t, w in raw.items()}
    return {t: round(w, 6) for t, w in raw.items() if abs(w) > 1e-9}


class Portfolio:
    def __init__(self, cash_usd: float = 100000.0,
                 positions: Optional[Dict[str, Dict[str, float]]] = None,
                 cost_bps: float = 1.0):
        self.cash = float(cash_usd)
        self.positions: Dict[str, Dict[str, float]] = {
            t.upper(): {"qty": float(p.get("qty", 0.0)), "avg_price": float(p.get("avg_price", 0.0))}
            for t, p in (positions or {}).items()
        }
        self.cost_bps = float(cost_bps)
        self.trade_log: List[Trade] = []

    def equity(self, prices: Dict[str, float]) -> float:
        total = self.cash
        for ticker, pos in self.positions.items():
            px = prices.get(ticker)
            if px:
                total += pos["qty"] * px
        return total

    def weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {}
        return {t: round(p["qty"] * prices.get(t, 0.0) / eq, 6)
                for t, p in self.positions.items() if p["qty"]}

    def rebalance_to(self, targets: Dict[str, float], prices: Dict[str, float]) -> List[Trade]:
        """Move to target weights at the given prices. Returns the fills."""
        equity = self.equity(prices)
        if equity <= 0:
            return []

        trades: List[Trade] = []
        tickers = set(targets) | set(self.positions)
        for ticker in sorted(tickers):
            price = prices.get(ticker)
            if not price or price <= 0:
                continue
            target_qty = targets.get(ticker, 0.0) * equity / price
            current_qty = self.positions.get(ticker, {}).get("qty", 0.0)
            delta = target_qty - current_qty
            if abs(delta * price) < max(1.0, equity * 1e-4):   # ignore dust
                continue

            notional = delta * price
            cost = abs(notional) * self.cost_bps / 10000.0
            self.cash -= notional + cost

            pos = self.positions.setdefault(ticker, {"qty": 0.0, "avg_price": 0.0})
            new_qty = pos["qty"] + delta
            if pos["qty"] == 0 or (pos["qty"] > 0) != (new_qty > 0):
                pos["avg_price"] = price
            elif abs(new_qty) > abs(pos["qty"]):                # adding to the position
                pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * delta) / new_qty
            pos["qty"] = new_qty
            if abs(pos["qty"]) < 1e-9:
                self.positions.pop(ticker, None)

            trade = Trade(ticker, round(delta, 6), round(price, 4),
                          round(notional, 2), round(cost, 4))
            trades.append(trade)
            self.trade_log.append(trade)
        return trades

    def apply_fill(self, ticker: str, signed_qty: float, price: float, cost: float) -> None:
        """Book one execution. `signed_qty` is positive for a buy."""
        ticker = ticker.upper()
        self.cash -= signed_qty * price + cost

        pos = self.positions.setdefault(ticker, {"qty": 0.0, "avg_price": 0.0})
        new_qty = pos["qty"] + signed_qty
        if pos["qty"] == 0 or (pos["qty"] > 0) != (new_qty > 0):
            pos["avg_price"] = price                      # opened or flipped
        elif abs(new_qty) > abs(pos["qty"]):
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * signed_qty) / new_qty
        pos["qty"] = new_qty
        if abs(pos["qty"]) < 1e-9:
            self.positions.pop(ticker, None)

        self.trade_log.append(Trade(ticker, round(signed_qty, 6), round(price, 4),
                                    round(signed_qty * price, 2), round(cost, 4)))

    def to_state(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """The `state.json` shape the agents read."""
        return {
            "cash_usd": round(self.cash, 2),
            "positions": {t: {"qty": round(p["qty"], 4), "avg_price": round(p["avg_price"], 4)}
                          for t, p in self.positions.items()},
            "prices": {t: round(px, 4) for t, px in prices.items()},
        }

    def snapshot(self, prices: Dict[str, float]) -> Dict[str, Any]:
        return {
            "equity": round(self.equity(prices), 2),
            "cash": round(self.cash, 2),
            "gross_exposure": round(sum(abs(w) for w in self.weights(prices).values()), 4),
            "net_exposure": round(sum(self.weights(prices).values()), 4),
            "weights": self.weights(prices),
        }
