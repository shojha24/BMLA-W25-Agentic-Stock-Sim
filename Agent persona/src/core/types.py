from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, TypedDict

Direction = Literal["UP", "DOWN", "FLAT"]
RiskRegime = Literal["RISK_ON", "RISK_OFF", "NEUTRAL"]


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


class Forecast(TypedDict):
    """The scorable unit of the simulation.

    Every agent must emit these; the consensus aggregates them and the
    evaluator scores them against realized returns.
    """
    ticker: str
    direction: Direction
    expected_return_bps: float    # signed, basis points over `horizon`
    horizon: str                  # "15m" | "1d" | "5d"
    confidence: float             # 0..1
    rationale: str
    news_refs: List[str]          # news_id / doc_id references


class Order(TypedDict):
    """An agent's actual trade instruction, not an opinion.

    The whiteboard's {"Actions": ["Buy/Sell", "#Shares", "stock"]}, typed.
    """
    ticker: str
    side: Literal["BUY", "SELL"]
    qty: float                    # shares, always positive; `side` carries direction
    order_type: Literal["MARKET", "LIMIT"]
    limit_price: Optional[float]
    rationale: str
    news_refs: List[str]


class Fill(TypedDict):
    """What the market simulator did with an order."""
    agent_id: str
    timestamp: str
    ticker: str
    side: str
    requested_qty: float
    filled_qty: float
    price: float                  # execution price, after slippage
    notional: float               # signed: negative = cash out
    cost: float
    status: Literal["FILLED", "PARTIAL", "REJECTED"]
    reason: str


class AgentOutput(TypedDict):
    agent_name: str
    persona: str
    timestamp: str
    decision: str
    market_view: Dict[str, Any]
    signals: List[Dict[str, Any]]
    forecasts: List[Forecast]
    orders: List[Order]
    trade_ideas: List[Dict[str, Any]]
    checks: Dict[str, Any]


class ConsensusForecast(TypedDict):
    ticker: str
    direction: Direction
    expected_return_bps: float
    horizon: str
    confidence: float             # 0..1, already discounted by disagreement
    agreement: float              # 0..1, weighted share of the modal direction
    net_vote: float               # -1..1, weighted directional vote (UP=+1, DOWN=-1)
    dispersion_bps: float         # stdev of member expected_return_bps
    n_agents: int
    votes: List[Dict[str, Any]]   # per-agent direction/bps/confidence


class ConsensusResult(TypedDict):
    timestamp: str
    horizon: str
    risk_regime: RiskRegime
    forecasts: List[ConsensusForecast]
    mean_agreement: float
    agents: List[str]
