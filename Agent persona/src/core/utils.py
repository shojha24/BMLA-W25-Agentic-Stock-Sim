from __future__ import annotations
from typing import Dict, Any


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def estimate_equity(state: Dict[str, Any]) -> float:
    cash = float(state.get("cash_usd", 0.0))
    positions = state.get("positions", {}) or {}
    prices = state.get("prices", {}) or {}
    equity = cash
    for t, pos in positions.items():
        qty = int(pos.get("qty", 0))
        px = float(prices.get(t, 0.0))
        equity += qty * px
    return equity