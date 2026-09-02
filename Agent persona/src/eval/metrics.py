"""Scoring. Forecast quality first, then portfolio outcome.

A model can make money by accident, so directional accuracy and calibration are
reported next to returns rather than instead of them.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Sequence

TRADING_DAYS = 252
MIN_PERIODS_FOR_ANNUALIZING = 60   # ~3 months; below this, annualized figures are noise


def score_forecasts(forecasts: Sequence[Dict[str, Any]], realized_bps: Dict[str, float],
                    flat_band_bps: float = 10.0) -> List[Dict[str, Any]]:
    """Attach the realized move to each forecast and mark it right or wrong."""
    scored = []
    for f in forecasts:
        ticker = str(f.get("ticker", "")).upper()
        actual = realized_bps.get(ticker)
        if actual is None:
            continue
        predicted = str(f.get("direction", "FLAT")).upper()
        actual_dir = "FLAT" if abs(actual) <= flat_band_bps else ("UP" if actual > 0 else "DOWN")
        directional = predicted in ("UP", "DOWN")
        scored.append({
            "ticker": ticker,
            "direction": predicted,
            "expected_return_bps": float(f.get("expected_return_bps", 0.0)),
            "confidence": float(f.get("confidence", 0.0)),
            "realized_bps": round(actual, 2),
            "realized_direction": actual_dir,
            "correct": (predicted == actual_dir) if directional else None,
            # A directional call is "right on sign" whenever it made money, even
            # if the move was small enough to count as FLAT.
            "sign_correct": (actual > 0) == (predicted == "UP") if directional else None,
            "abs_error_bps": round(abs(float(f.get("expected_return_bps", 0.0)) - actual), 2),
        })
    return scored


def summarize_forecasts(scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    directional = [s for s in scored if s["sign_correct"] is not None]
    hits = [1.0 if s["sign_correct"] else 0.0 for s in directional]

    # Brier score on the directional call: forecast probability = confidence.
    brier = [
        (s["confidence"] - (1.0 if s["sign_correct"] else 0.0)) ** 2
        for s in directional
    ]
    # Confidence-weighted accuracy: does the model know when it knows?
    weight = sum(s["confidence"] for s in directional)
    weighted_hits = (
        sum(s["confidence"] * (1.0 if s["sign_correct"] else 0.0) for s in directional) / weight
        if weight else 0.0
    )
    return {
        "n_forecasts": len(scored),
        "n_directional": len(directional),
        "hit_rate": round(sum(hits) / len(hits), 4) if hits else None,
        "confidence_weighted_hit_rate": round(weighted_hits, 4) if directional else None,
        "brier_score": round(sum(brier) / len(brier), 4) if brier else None,
        "mean_abs_error_bps": round(statistics.fmean(s["abs_error_bps"] for s in scored), 2) if scored else None,
        "mean_realized_bps": round(statistics.fmean(s["realized_bps"] for s in scored), 2) if scored else None,
        "flat_share": round(sum(1 for s in scored if s["direction"] == "FLAT") / len(scored), 4) if scored else None,
    }


def returns_from_equity(curve: Sequence[float]) -> List[float]:
    return [curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve)) if curve[i - 1]]


def summarize_equity(curve: Sequence[float], periods_per_year: int = TRADING_DAYS) -> Dict[str, Any]:
    if len(curve) < 2:
        return {"cumulative_return": 0.0, "n_periods": len(curve)}
    rets = returns_from_equity(curve)
    if not rets:
        return {"cumulative_return": 0.0, "n_periods": len(curve)}

    mean = statistics.fmean(rets)
    sd = statistics.pstdev(rets)
    peak, max_dd = curve[0], 0.0
    for value in curve:
        peak = max(peak, value)
        if peak:
            max_dd = min(max_dd, value / peak - 1.0)
    total = curve[-1] / curve[0] - 1.0
    years = len(rets) / periods_per_year
    # Annualizing a two-week sample produces numbers like +3678%; report it only
    # when the window is long enough for the figure to mean anything.
    annualized = (
        round((1 + total) ** (1 / years) - 1, 6)
        if len(rets) >= MIN_PERIODS_FOR_ANNUALIZING and years > 0 and (1 + total) > 0 else None
    )
    return {
        "cumulative_return": round(total, 6),
        "annualized_return": annualized,
        "annualized_vol": round(sd * math.sqrt(periods_per_year), 6),
        "sharpe": round(mean / sd * math.sqrt(periods_per_year), 4) if sd > 0 else None,
        "max_drawdown": round(max_dd, 6),
        "hit_days": round(sum(1 for r in rets if r > 0) / len(rets), 4),
        "n_periods": len(curve),
    }


def compare(models: Dict[str, Dict[str, Any]], key: str = "sharpe") -> List[Dict[str, Any]]:
    rows = [{"model": name, **stats} for name, stats in models.items()]
    rows.sort(key=lambda r: (r.get(key) is None, -(r.get(key) or 0)))
    return rows


REGIME_SIGN = {"RISK_ON": 1, "RISK_OFF": -1, "NEUTRAL": 0}


def summarize_sim_vs_actual(cycles: Sequence[Dict[str, Any]],
                            flat_band_bps: float = 10.0) -> Dict[str, Any]:
    """The whiteboard's last box: the simulation's collective call vs the market.

    Three questions, one per row of the result:
      * did the panel's risk regime match what the index actually did?
      * was the consensus directionally right, per ticker?
      * was it biased - did it expect bigger or smaller moves than happened?
    """
    regime_calls = regime_hits = 0
    directional = hits = 0
    expected: List[float] = []
    actual: List[float] = []
    scored_cycles = 0

    for cycle in cycles:
        consensus = cycle.get("consensus") or {}
        realized = cycle.get("realized_bps") or {}
        if not realized:
            continue
        scored_cycles += 1

        index_move = cycle.get("index_bps")
        regime = str(consensus.get("risk_regime", "NEUTRAL")).upper()
        if index_move is not None and regime in REGIME_SIGN:
            called = REGIME_SIGN[regime]
            happened = 0 if abs(index_move) <= flat_band_bps else (1 if index_move > 0 else -1)
            regime_calls += 1
            regime_hits += int(called == happened)

        for f in consensus.get("forecasts", []) or []:
            moved = realized.get(str(f.get("ticker", "")).upper())
            if moved is None:
                continue
            expected.append(float(f.get("expected_return_bps", 0.0)))
            actual.append(moved)
            if str(f.get("direction")) in ("UP", "DOWN"):
                directional += 1
                hits += int((moved > 0) == (f["direction"] == "UP"))

    mean_expected = statistics.fmean(expected) if expected else 0.0
    mean_actual = statistics.fmean(actual) if actual else 0.0
    return {
        "cycles_scored": scored_cycles,
        "regime_calls": regime_calls,
        "regime_accuracy": round(regime_hits / regime_calls, 4) if regime_calls else None,
        "directional_calls": directional,
        "directional_accuracy": round(hits / directional, 4) if directional else None,
        "mean_expected_bps": round(mean_expected, 2),
        "mean_actual_bps": round(mean_actual, 2),
        # Positive = the panel expected more movement than the market delivered.
        "magnitude_bias_bps": round(mean_expected - mean_actual, 2),
    }
