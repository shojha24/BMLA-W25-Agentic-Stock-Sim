"""Naive models the agent panel has to beat.

If the panel cannot beat "assume today's move continues" or "always long", the
LLMs are decoration. Each forecaster emits the same Forecast records as an
agent, so they run through identical sizing, costs and scoring.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from agents.baseline import SentimentBaselineAgent
from core.types import Forecast


class NaiveForecaster(ABC):
    name: str = "naive"

    @abstractmethod
    def forecast(self, *, digest: Dict[str, Any], state: Dict[str, Any], universe: Sequence[str],
                 day: str, prices: Any, horizon: str) -> List[Forecast]:
        ...

    @staticmethod
    def _mk(ticker: str, direction: str, bps: float, conf: float, horizon: str,
            why: str) -> Forecast:
        return {
            "ticker": ticker.upper(), "direction": direction,  # type: ignore[typeddict-item]
            "expected_return_bps": round(bps, 2), "horizon": horizon,
            "confidence": round(conf, 3), "rationale": why, "news_refs": [],
        }


class AlwaysLongForecaster(NaiveForecaster):
    """Equity drift. The hardest naive model to beat in a bull sample."""

    name = "always_long"

    def forecast(self, *, digest, state, universe, day, prices, horizon) -> List[Forecast]:
        return [self._mk(t, "UP", 20.0, 0.5, horizon, "always long") for t in universe]


class RandomForecaster(NaiveForecaster):
    """Coin flips - the zero-skill reference."""

    name = "random"

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def forecast(self, *, digest, state, universe, day, prices, horizon) -> List[Forecast]:
        out = []
        for t in universe:
            up = self.rng.random() > 0.5
            out.append(self._mk(t, "UP" if up else "DOWN", 20.0 if up else -20.0, 0.5,
                                horizon, "coin flip"))
        return out


class PersistenceForecaster(NaiveForecaster):
    """Yesterday's move repeats - the classic naive time-series baseline."""

    name = "persistence"

    def __init__(self, lookback_sessions: int = 1):
        self.lookback = lookback_sessions

    def forecast(self, *, digest, state, universe, day, prices, horizon) -> List[Forecast]:
        out = []
        for t in universe:
            prev_day = prices.next_session(t, day, -self.lookback)
            last = prices.close_on(t, day)
            prior = prices.close_on(t, prev_day) if prev_day else None
            if not last or not prior or prior <= 0:
                out.append(self._mk(t, "FLAT", 0.0, 0.2, horizon, "no prior close"))
                continue
            move_bps = (last / prior - 1.0) * 10000.0
            direction = "FLAT" if abs(move_bps) <= 10 else ("UP" if move_bps > 0 else "DOWN")
            out.append(self._mk(t, direction, move_bps * 0.5, min(0.3 + abs(move_bps) / 400.0, 0.8),
                                horizon, f"prior session {move_bps:+.0f}bps"))
        return out


class ReversalForecaster(NaiveForecaster):
    """The mirror of persistence - one of the two has to look good."""

    name = "reversal"

    def forecast(self, *, digest, state, universe, day, prices, horizon) -> List[Forecast]:
        flipped = []
        for f in PersistenceForecaster().forecast(digest=digest, state=state, universe=universe,
                                                  day=day, prices=prices, horizon=horizon):
            direction = {"UP": "DOWN", "DOWN": "UP", "FLAT": "FLAT"}[f["direction"]]
            flipped.append(self._mk(f["ticker"], direction, -f["expected_return_bps"],
                                    f["confidence"], horizon, "fade prior session"))
        return flipped


class SentimentRuleForecaster(NaiveForecaster):
    """Read the digest sentiment, skip the LLM. The 'is the LLM earning its keep?' control."""

    name = "sentiment_rule"

    def __init__(self):
        self.agent = SentimentBaselineAgent()

    def forecast(self, *, digest, state, universe, day, prices, horizon) -> List[Forecast]:
        self.agent.horizon = horizon
        out = self.agent.run(digest, state)
        return [f for f in out["forecasts"] if f["ticker"].upper() in {u.upper() for u in universe}]


DEFAULT_BENCHMARKS = ["always_long", "random", "persistence", "reversal", "sentiment_rule"]

_REGISTRY = {
    "always_long": AlwaysLongForecaster,
    "random": RandomForecaster,
    "persistence": PersistenceForecaster,
    "reversal": ReversalForecaster,
    "sentiment_rule": SentimentRuleForecaster,
}


def build_benchmarks(names: Optional[Sequence[str]] = None) -> List[NaiveForecaster]:
    chosen = list(names if names is not None else DEFAULT_BENCHMARKS)
    unknown = [n for n in chosen if n not in _REGISTRY]
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {unknown}. Available: {sorted(_REGISTRY)}")
    return [_REGISTRY[n]() for n in chosen]
