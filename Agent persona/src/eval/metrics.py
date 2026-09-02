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
