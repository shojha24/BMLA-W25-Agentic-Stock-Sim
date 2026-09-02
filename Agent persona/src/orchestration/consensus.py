"""Aggregate per-agent forecasts into one consensus forecast.

Confidence-weighted, and explicitly discounted by disagreement: three agents
that split 2-1 on direction should not produce the same consensus confidence as
three that agree. Dispersion is reported, not hidden, because agent
disagreement is itself a signal worth evaluating.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional

from core.types import AgentOutput, ConsensusForecast, ConsensusResult

DIRECTION_VOTE = {"UP": 1.0, "DOWN": -1.0, "FLAT": 0.0}
FLAT_SCORE_BAND = 0.15      # |net weighted vote| below this is not a directional call
FLAT_BPS_BAND = 10.0
MIN_WEIGHT = 1e-6


def _weight(forecast: Dict[str, Any], agent_weight: float) -> float:
    return max(float(forecast.get("confidence", 0.0)), 0.0) * agent_weight


def aggregate_ticker(votes: List[Dict[str, Any]], horizon: str) -> ConsensusForecast:
    total_w = sum(v["weight"] for v in votes)
    if total_w <= MIN_WEIGHT:
        # Every member abstained with zero confidence: fall back to a flat, unweighted read.
        total_w = float(len(votes))
        for v in votes:
            v["weight"] = 1.0

    score = sum(v["weight"] * DIRECTION_VOTE.get(v["direction"], 0.0) for v in votes) / total_w
    bps = sum(v["weight"] * v["expected_return_bps"] for v in votes) / total_w

    # Agreement is the weighted share of the modal call, so a unanimous FLAT
    # ("nothing happens here") counts as agreement, not as a coin flip.
    share: Dict[str, float] = {}
    for v in votes:
        share[v["direction"]] = share.get(v["direction"], 0.0) + v["weight"]
    agreement = max(share.values()) / total_w

    if score > FLAT_SCORE_BAND and bps > FLAT_BPS_BAND:
        direction = "UP"
    elif score < -FLAT_SCORE_BAND and bps < -FLAT_BPS_BAND:
        direction = "DOWN"
    else:
        direction = "FLAT"
        bps = max(min(bps, FLAT_BPS_BAND), -FLAT_BPS_BAND)

    mean_conf = sum(v["weight"] * v["confidence"] for v in votes) / total_w
    dispersion = statistics.pstdev([v["expected_return_bps"] for v in votes]) if len(votes) > 1 else 0.0

    return {
        "ticker": votes[0]["ticker"],
        "direction": direction,  # type: ignore[typeddict-item]
        "expected_return_bps": round(bps, 2),
        "horizon": horizon,
        "confidence": round(mean_conf * agreement, 4),
        "agreement": round(agreement, 4),
        "net_vote": round(score, 4),
        "dispersion_bps": round(dispersion, 2),
        "n_agents": len(votes),
        "votes": [
            {k: v[k] for k in ("agent_name", "direction", "expected_return_bps", "confidence")}
            for v in votes
        ],
    }


def build_consensus(
    outputs: Iterable[AgentOutput],
    timestamp: str = "",
    horizon: str = "1d",
    agent_weights: Optional[Dict[str, float]] = None,
) -> ConsensusResult:
    outputs = list(outputs)
    agent_weights = agent_weights or {}

    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for out in outputs:
        agent = str(out.get("agent_name", "unknown"))
        aw = float(agent_weights.get(agent, 1.0))
        for f in out.get("forecasts", []) or []:
            ticker = str(f.get("ticker", "")).upper()
            if not ticker:
                continue
            by_ticker.setdefault(ticker, []).append({
                "agent_name": agent,
                "ticker": ticker,
                "direction": str(f.get("direction", "FLAT")).upper(),
                "expected_return_bps": float(f.get("expected_return_bps", 0.0)),
                "confidence": float(f.get("confidence", 0.0)),
                "weight": _weight(f, aw),
            })

    forecasts = [aggregate_ticker(v, horizon) for v in by_ticker.values()]
    forecasts.sort(key=lambda f: (-f["confidence"], f["ticker"]))

    # Regime vote, weighted by each agent's own stated confidence in its view.
    regime_scores: Dict[str, float] = {}
    for out in outputs:
        mv = out.get("market_view") or {}
        regime = str(mv.get("risk_regime", "NEUTRAL")).upper()
        regime_scores[regime] = regime_scores.get(regime, 0.0) + float(mv.get("confidence", 0.5) or 0.0)
    regime = max(regime_scores, key=regime_scores.get) if regime_scores else "NEUTRAL"

    mean_agreement = (
        sum(f["agreement"] for f in forecasts) / len(forecasts) if forecasts else 0.0
    )

    return {
        "timestamp": timestamp or (outputs[0].get("timestamp", "") if outputs else ""),
        "horizon": horizon,
        "risk_regime": regime,  # type: ignore[typeddict-item]
        "forecasts": forecasts,
        "mean_agreement": round(mean_agreement, 4),
        "agents": [str(o.get("agent_name", "unknown")) for o in outputs],
    }
