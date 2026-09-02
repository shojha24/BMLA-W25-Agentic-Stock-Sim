"""Coercion + validation for LLM output.

LLMs return "mostly right" JSON: strings where floats belong, tickers that were
never in the universe, confidences of 95 instead of 0.95. Everything downstream
(consensus, scoring, the simulator) assumes clean records, so all model output
funnels through `normalize_agent_output` first.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from core.types import AgentOutput, Forecast, Order

VALID_DIRECTIONS = ("UP", "DOWN", "FLAT")
VALID_REGIMES = ("RISK_ON", "RISK_OFF", "NEUTRAL")
MAX_ABS_BPS = 2000.0        # +/- 20% over one horizon is already absurd
FLAT_BAND_BPS = 10.0        # |expected move| under this is FLAT by definition
MAX_FORECASTS = 12
MAX_ORDERS = 8
MAX_SIGNALS = 8
MAX_TRADE_IDEAS = 6

_SIDE_ALIASES = {
    "BUY": "BUY", "B": "BUY", "LONG": "BUY", "COVER": "BUY", "ADD": "BUY",
    "SELL": "SELL", "S": "SELL", "SHORT": "SELL", "REDUCE": "SELL", "EXIT": "SELL",
}
_NO_TRADE = {"HOLD", "NONE", "NOOP", "PASS", "FLAT", "WAIT", "NO_ACTION"}

_DIRECTION_ALIASES = {
    "UP": "UP", "BULLISH": "UP", "LONG": "UP", "OVERWEIGHT": "UP", "BUY": "UP",
    "POSITIVE": "UP", "+": "UP",
    "DOWN": "DOWN", "BEARISH": "DOWN", "SHORT": "DOWN", "UNDERWEIGHT": "DOWN",
    "SELL": "DOWN", "NEGATIVE": "DOWN", "-": "DOWN",
    "FLAT": "FLAT", "NEUTRAL": "FLAT", "HOLD": "FLAT", "UNCHANGED": "FLAT",
}


class SchemaError(ValueError):
    pass


def extract_json(text: str) -> Dict[str, Any]:
    """Pull a JSON object out of a model response.

    Handles bare JSON, ```json fences, and prose wrapped around an object.
    """
    if not isinstance(text, str) or not text.strip():
        raise SchemaError("Model returned empty content.")
    s = text.strip()

    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()

    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j <= i:
        raise SchemaError(f"Model did not return JSON. First 200 chars: {text[:200]!r}")
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Model returned malformed JSON: {exc}") from exc


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if m:
            return float(m.group(0))
    return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_confidence(value: Any, default: float = 0.5) -> float:
    """Land any of the scales models actually emit on 0..1.

    0..1 as asked; 1..10 read as a ten-point scale ("confidence: 8");
    10..100 read as a percentage ("80", "65%"); anything above that is clamped.
    """
    c = as_float(value, default)
    if c > 100.0:
        c = 100.0
    if c > 10.0:
        c = c / 100.0
    elif c > 1.0:
        c = c / 10.0
    return clamp(c, 0.0, 1.0)


def normalize_direction(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().upper()
    return _DIRECTION_ALIASES.get(key)


def normalize_regime(value: Any) -> str:
    key = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    return key if key in VALID_REGIMES else "NEUTRAL"


def _as_str_list(value: Any, limit: int = 12) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable):
        return []
    return [str(v) for v in list(value)[:limit] if str(v).strip()]


def normalize_forecast(
    raw: Any,
    universe: Optional[Iterable[str]],
    default_horizon: str,
) -> Optional[Forecast]:
    if not isinstance(raw, dict):
        return None

    ticker = str(raw.get("ticker") or raw.get("symbol") or "").strip().upper()
    if not ticker:
        return None
    if universe is not None and ticker not in {t.upper() for t in universe}:
        return None  # hallucinated ticker: the agent may only trade what it can price

    bps = clamp(as_float(raw.get("expected_return_bps"), 0.0), -MAX_ABS_BPS, MAX_ABS_BPS)
    direction = normalize_direction(raw.get("direction"))
    if direction is None:
        # Infer from the magnitude when the model omitted or mangled the label.
        direction = "FLAT" if abs(bps) <= FLAT_BAND_BPS else ("UP" if bps > 0 else "DOWN")

    # Direction is the primary claim; make the magnitude agree with it.
    if direction == "FLAT":
        bps = clamp(bps, -FLAT_BAND_BPS, FLAT_BAND_BPS)
    elif direction == "UP" and bps <= 0:
        bps = max(abs(bps), FLAT_BAND_BPS + 1.0)
    elif direction == "DOWN" and bps >= 0:
        bps = -max(abs(bps), FLAT_BAND_BPS + 1.0)

    return {
        "ticker": ticker,
        "direction": direction,  # type: ignore[typeddict-item]
        "expected_return_bps": round(bps, 2),
        "horizon": str(raw.get("horizon") or default_horizon),
        "confidence": round(normalize_confidence(raw.get("confidence")), 4),
        "rationale": str(raw.get("rationale") or "")[:600],
        "news_refs": _as_str_list(raw.get("news_refs") or raw.get("refs")),
    }


def normalize_forecasts(
    raw_list: Any,
    universe: Optional[Iterable[str]] = None,
    default_horizon: str = "1d",
) -> List[Forecast]:
    """Clean, de-duplicate (highest confidence wins per ticker) and cap."""
    if not isinstance(raw_list, list):
        return []
    by_ticker: Dict[str, Forecast] = {}
    for raw in raw_list:
        f = normalize_forecast(raw, universe, default_horizon)
        if f is None:
            continue
        prev = by_ticker.get(f["ticker"])
        if prev is None or f["confidence"] > prev["confidence"]:
            by_ticker[f["ticker"]] = f
    out = sorted(by_ticker.values(), key=lambda f: -f["confidence"])
    return out[:MAX_FORECASTS]


def normalize_order(raw: Any, universe: Optional[Iterable[str]]) -> Optional[Order]:
    """Coerce one action into an Order, or None if it is not a trade.

    Accepts the dict form and the whiteboard's positional form
    ["Buy", 10, "AAPL"], in any argument order.
    """
    if isinstance(raw, (list, tuple)):
        raw = _order_from_sequence(raw)
    if not isinstance(raw, dict):
        return None

    side_raw = raw.get("side") or raw.get("action") or raw.get("buy_sell") or ""
    side_key = str(side_raw).strip().upper()
    if side_key in _NO_TRADE:
        return None                       # "no action" is a legal answer
    side = _SIDE_ALIASES.get(side_key)
    if side is None:
        return None

    ticker = str(raw.get("ticker") or raw.get("stock") or raw.get("symbol") or "").strip().upper()
    if not ticker:
        return None
    if universe is not None and ticker not in {t.upper() for t in universe}:
        return None                       # cannot trade what the simulator cannot price

    qty = abs(as_float(raw.get("qty") if raw.get("qty") is not None else
                       raw.get("shares") or raw.get("quantity") or raw.get("num_shares"), 0.0))
    if qty <= 0:
        return None

    order_type = str(raw.get("order_type") or "MARKET").strip().upper()
    limit_price_raw = raw.get("limit_price")
    limit_price = as_float(limit_price_raw, 0.0) if limit_price_raw is not None else None
    if order_type != "LIMIT" or not limit_price or limit_price <= 0:
        order_type, limit_price = "MARKET", None

    return {
        "ticker": ticker,
        "side": side,  # type: ignore[typeddict-item]
        "qty": round(qty, 4),
        "order_type": order_type,  # type: ignore[typeddict-item]
        "limit_price": round(limit_price, 4) if limit_price else None,
        "rationale": str(raw.get("rationale") or "")[:400],
        "news_refs": _as_str_list(raw.get("news_refs") or raw.get("refs")),
    }


def _order_from_sequence(seq: Iterable[Any]) -> Dict[str, Any]:
    """["Buy", 10, "AAPL"] -> {side, qty, ticker}, order-insensitive."""
    out: Dict[str, Any] = {}
    for part in list(seq)[:4]:
        text = str(part).strip()
        upper = text.upper()
        if upper in _SIDE_ALIASES or upper in _NO_TRADE:
            out["side"] = upper
        elif re.fullmatch(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
            out["qty"] = text
        elif re.fullmatch(r"[A-Za-z.\-]{1,6}", text):
            out["ticker"] = upper
    return out


def normalize_orders(raw_list: Any, universe: Optional[Iterable[str]] = None) -> List[Order]:
    """Clean a batch of actions, merging duplicates on the same ticker and side."""
    if raw_list is None:
        return []
    if isinstance(raw_list, dict):
        raw_list = [raw_list]
    if not isinstance(raw_list, list):
        return []

    merged: Dict[tuple, Order] = {}
    for raw in raw_list:
        order = normalize_order(raw, universe)
        if order is None:
            continue
        key = (order["ticker"], order["side"], order["order_type"])
        if key in merged:
            merged[key]["qty"] = round(merged[key]["qty"] + order["qty"], 4)
        else:
            merged[key] = order
    return list(merged.values())[:MAX_ORDERS]


def normalize_agent_output(
    raw: Dict[str, Any],
    *,
    agent_name: str,
    persona: str,
    timestamp: str,
    universe: Optional[Iterable[str]] = None,
    default_horizon: str = "1d",
) -> AgentOutput:
    if not isinstance(raw, dict):
        raise SchemaError("Agent output was not a JSON object.")

    market_view = raw.get("market_view")
    if not isinstance(market_view, dict):
        market_view = {}
    market_view["risk_regime"] = normalize_regime(market_view.get("risk_regime"))
    market_view["confidence"] = round(normalize_confidence(market_view.get("confidence")), 4)
    market_view["summary"] = str(market_view.get("summary") or "")[:1000]
    market_view["key_drivers"] = _as_str_list(market_view.get("key_drivers"))

    signals = raw.get("signals")
    signals = signals[:MAX_SIGNALS] if isinstance(signals, list) else []
    for s in signals:
        if isinstance(s, dict):
            s["confidence"] = round(normalize_confidence(s.get("confidence")), 4)

    trade_ideas = raw.get("trade_ideas")
    trade_ideas = trade_ideas[:MAX_TRADE_IDEAS] if isinstance(trade_ideas, list) else []

    out: AgentOutput = {
        "agent_name": str(raw.get("agent_name") or agent_name),
        "persona": str(raw.get("persona") or persona),
        "timestamp": str(raw.get("timestamp") or timestamp),
        "decision": str(raw.get("decision") or "")[:600],
        "market_view": market_view,
        "signals": signals,
        "forecasts": normalize_forecasts(raw.get("forecasts"), universe, default_horizon),
        # The whiteboard calls these "Actions"; accept either key.
        "orders": normalize_orders(raw.get("orders") if raw.get("orders") is not None
                                   else raw.get("actions"), universe),
        "trade_ideas": trade_ideas,
        "checks": raw.get("checks") if isinstance(raw.get("checks"), dict) else {},
    }
    return out
