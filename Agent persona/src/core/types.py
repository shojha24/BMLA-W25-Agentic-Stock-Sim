from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict


class DigestItem(TypedDict):
    news_id: str
    time: str
    source: str
    headline: str
    summary: str
    tickers_mentioned: List[str]
    macro_tags: List[str]         # e.g. ["CPI", "YIELDS", "RISK_OFF"]
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence: float             # 0..1


class Digest(TypedDict):
    timestamp: str
    news_digest: List[DigestItem]


class Position(TypedDict):
    qty: int
    avg_price: float


class State(TypedDict):
    cash_usd: float
    positions: Dict[str, Position]
    prices: Dict[str, float]


class AgentOutput(TypedDict):
    agent_name: str
    persona: str
    timestamp: str
    decision: str
    market_view: Dict[str, Any]
    signals: List[Dict[str, Any]]
    trade_ideas: List[Dict[str, Any]]
    checks: Dict[str, Any]