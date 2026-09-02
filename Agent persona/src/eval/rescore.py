"""Score a finished run against prices that did not exist when it ran.

Live cycles cannot be scored as they happen: the forecast is for the next 15
minutes or the next session, and that has not occurred yet. Every cycle is
written to the run log, so scoring is a separate pass - fetch the prices that
have since printed, and judge what was already recorded.

This also lets a backtest be re-scored on a different horizon without paying
for the LLM calls again.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from data.market_data import (PriceStore, horizon_to_bars, horizon_to_sessions, is_intraday)
from eval.metrics import score_forecasts, summarize_forecasts, summarize_sim_vs_actual


def load_cycles(log_path: Path | str) -> List[Dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"No run log at {path}")
    cycles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cycles.append(json.loads(line))
    return cycles


def _models(cycle: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out = {"agents_consensus": (cycle.get("consensus") or {}).get("forecasts", []) or []}
    round1 = (cycle.get("consensus_round1") or {}).get("forecasts")
    if round1:
        out["agents_round1"] = round1
    for agent in cycle.get("agent_outputs", []) or []:
        out[f"agent:{agent.get('agent_name', 'unknown')}"] = agent.get("forecasts", []) or []
    return out


def rescore_run(
    log_path: Path | str,
    prices: Optional[PriceStore] = None,
    horizon: str = "",
    universe: Optional[Iterable[str]] = None,
    index_ticker: str = "SPY",
    bar_interval: str = "15m",
    refresh: bool = False,
) -> Dict[str, Any]:
    """Re-score every cycle in a run log. Returns a report shaped like the engine's."""
    cycles = load_cycles(log_path)
    if not cycles:
        raise ValueError(f"Run log {log_path} has no cycles")
    prices = prices or PriceStore()

    horizon = horizon or str((cycles[0].get("consensus") or {}).get("horizon") or "1d")
    tickers = sorted({t for c in cycles for t in (c.get("prices") or {})} |
                     set(universe or ()) | {index_ticker})
    intraday = is_intraday(horizon)

    days = sorted({c.get("day", "")[:10] for c in cycles if c.get("day")})
    if intraday:
        prices.ensure_intraday(tickers, bar_interval, refresh=refresh)
    else:
        prices.ensure(tickers, days[0] if days else "2020-01-01",
                      days[-1] if days else "2020-12-31", refresh=refresh)

    bars = horizon_to_bars(horizon, 15)
    sessions = horizon_to_sessions(horizon)

    scored: Dict[str, List[Dict[str, Any]]] = {}
    enriched: List[Dict[str, Any]] = []
    unscored = 0

    for cycle in cycles:
        day = str(cycle.get("day", ""))[:10]
        timestamp = str(cycle.get("timestamp", ""))
        realized: Dict[str, float] = {}
        for ticker in (cycle.get("prices") or {}):
            if ticker == index_ticker:
                continue
            value = (prices.forward_return_bps_intraday(ticker, timestamp, bars, bar_interval)
                     if intraday else prices.forward_return_bps(ticker, day, sessions))
            if value is not None:
                realized[ticker] = value

        if not realized:
            unscored += 1
            continue

        index_bps = (prices.forward_return_bps_intraday(index_ticker, timestamp, bars, bar_interval)
                     if intraday else prices.forward_return_bps(index_ticker, day, sessions))
        for name, forecasts in _models(cycle).items():
            scored.setdefault(name, []).extend(score_forecasts(forecasts, realized))
        enriched.append({"day": day, "timestamp": timestamp,
                         "consensus": cycle.get("consensus") or {},
                         "realized_bps": realized, "index_bps": index_bps})

    return {
        "run_log": str(log_path),
        "run_id": cycles[0].get("run_id", ""),
        "horizon": horizon,
        "bar_interval": bar_interval if intraday else "1d",
        "cycles_in_log": len(cycles),
        "cycles_scored": len(enriched),
        "cycles_not_yet_scorable": unscored,
        "forecast_metrics": {name: summarize_forecasts(rows) for name, rows in scored.items() if rows},
        "sim_vs_actual": summarize_sim_vs_actual(enriched),
    }
