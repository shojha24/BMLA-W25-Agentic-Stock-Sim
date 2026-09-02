"""Two-round agent roundtable.

Round 1: every persona forecasts independently from the same digest.
Round 2: every persona sees the others' forecasts and reasoning, and revises.

Both rounds are kept, along with the per-agent revision deltas, so the
"communication" step can be ablated and measured rather than asserted.
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Optional, Sequence

from agents.base import BaseAgent
from core.types import AgentOutput, ConsensusResult, Digest, State
from orchestration.consensus import build_consensus


def _run_one(
    agent: BaseAgent,
    digest: Digest,
    state: State,
    peer_context: Optional[List[Dict[str, Any]]],
    prior_output: Optional[AgentOutput],
    brief: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        out = agent.run(digest, state, peer_context=peer_context, prior_output=prior_output,
                        brief=brief)
        return {"ok": True, "agent": agent.name, "output": out, "error": ""}
    except Exception as exc:  # a flaky free-tier model must not kill the panel
        return {"ok": False, "agent": agent.name, "output": None, "error": f"{type(exc).__name__}: {exc}"}


def _forecast_index(output: AgentOutput) -> Dict[str, Dict[str, Any]]:
    return {str(f["ticker"]).upper(): f for f in (output.get("forecasts") or [])}


def revision_deltas(round1: List[AgentOutput], round2: List[AgentOutput]) -> Dict[str, Any]:
    """How much did talking to each other actually change?"""
    r1 = {str(o.get("agent_name")): _forecast_index(o) for o in round1}
    r2 = {str(o.get("agent_name")): _forecast_index(o) for o in round2}

    per_agent: Dict[str, Any] = {}
    flips = 0
    compared = 0
    bps_moves: List[float] = []
    conf_moves: List[float] = []

    for agent, before in r1.items():
        after = r2.get(agent, {})
        agent_flips, agent_bps, agent_conf, changed = 0, [], [], []
        for ticker, f1 in before.items():
            f2 = after.get(ticker)
            if not f2:
                continue
            compared += 1
            d_bps = float(f2["expected_return_bps"]) - float(f1["expected_return_bps"])
            d_conf = float(f2["confidence"]) - float(f1["confidence"])
            agent_bps.append(abs(d_bps))
            agent_conf.append(d_conf)
            bps_moves.append(abs(d_bps))
            conf_moves.append(d_conf)
            if f1["direction"] != f2["direction"]:
                agent_flips += 1
                flips += 1
                changed.append({
                    "ticker": ticker,
                    "from": f1["direction"],
                    "to": f2["direction"],
                    "delta_bps": round(d_bps, 2),
                })
        per_agent[agent] = {
            "direction_flips": agent_flips,
            "mean_abs_delta_bps": round(sum(agent_bps) / len(agent_bps), 2) if agent_bps else 0.0,
            "mean_delta_confidence": round(sum(agent_conf) / len(agent_conf), 4) if agent_conf else 0.0,
            "flipped": changed,
            "revision_note": (
                next((o.get("market_view", {}).get("revision_note", "")
                      for o in round2 if o.get("agent_name") == agent), "")
            ),
        }

    return {
        "compared_forecasts": compared,
        "direction_flips": flips,
        "flip_rate": round(flips / compared, 4) if compared else 0.0,
        "mean_abs_delta_bps": round(sum(bps_moves) / len(bps_moves), 2) if bps_moves else 0.0,
        "mean_delta_confidence": round(sum(conf_moves) / len(conf_moves), 4) if conf_moves else 0.0,
        "per_agent": per_agent,
    }


class Roundtable:
    def __init__(
        self,
        agents: Sequence[BaseAgent],
        rounds: int = 2,
        horizon: str = "1d",
        parallel: bool = True,
        agent_weights: Optional[Dict[str, float]] = None,
    ):
        if not agents:
            raise ValueError("Roundtable needs at least one agent.")
        self.agents = list(agents)
        self.rounds = max(1, rounds)
        self.horizon = horizon
        self.parallel = parallel
        self.agent_weights = agent_weights or {}

    def _run_round(
        self,
        digest: Digest,
        state: State,
        peer_by_agent: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        prior_by_agent: Optional[Dict[str, AgentOutput]] = None,
        state_by_agent: Optional[Dict[str, State]] = None,
        brief_by_agent: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        jobs = [
            (
                a,
                # Each agent trades its own book, so each sees its own balance sheet.
                (state_by_agent or {}).get(a.name, state),
                (peer_by_agent or {}).get(a.name),
                (prior_by_agent or {}).get(a.name),
                (brief_by_agent or {}).get(a.name),
            )
            for a in self.agents
        ]
        if not self.parallel or len(jobs) == 1:
            return [_run_one(a, digest, st, peers, prior, brief)
                    for a, st, peers, prior, brief in jobs]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(_run_one, a, digest, st, peers, prior, brief)
                       for a, st, peers, prior, brief in jobs]
            return [f.result() for f in futures]

    def run(self, digest: Digest, state: State,
            state_by_agent: Optional[Dict[str, State]] = None,
            brief_by_agent: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """`state_by_agent` gives each agent its own book; `brief_by_agent` its brief."""
        timestamp = str(digest.get("timestamp", ""))
        rounds: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        results = self._run_round(digest, state, state_by_agent=state_by_agent,
                                  brief_by_agent=brief_by_agent)
        outputs = [r["output"] for r in results if r["ok"]]
        errors += [{"agent": r["agent"], "error": r["error"]} for r in results if not r["ok"]]
        if not outputs:
            raise RuntimeError(f"All agents failed in round 1: {errors}")

        rounds.append({
            "round": 1,
            "outputs": outputs,
            "consensus": build_consensus(outputs, timestamp, self.horizon, self.agent_weights),
        })

        for round_no in range(2, self.rounds + 1):
            prev = rounds[-1]["outputs"]
            prior_by_agent = {str(o.get("agent_name")): o for o in prev}
            # Each agent sees everyone's view but its own.
            peer_by_agent = {
                a.name: [o for o in prev if str(o.get("agent_name")) != a.name]
                for a in self.agents
            }
            results = self._run_round(digest, state, peer_by_agent, prior_by_agent,
                                      state_by_agent=state_by_agent,
                                      brief_by_agent=brief_by_agent)
            outputs = [r["output"] for r in results if r["ok"]]
            errors += [{"agent": r["agent"], "error": r["error"]} for r in results if not r["ok"]]
            if not outputs:
                break  # keep the last good round rather than failing the cycle
            rounds.append({
                "round": round_no,
                "outputs": outputs,
                "consensus": build_consensus(outputs, timestamp, self.horizon, self.agent_weights),
            })

        final: ConsensusResult = rounds[-1]["consensus"]
        deltas = (
            revision_deltas(rounds[0]["outputs"], rounds[-1]["outputs"])
            if len(rounds) > 1 else {}
        )

        return {
            "timestamp": timestamp,
            "horizon": self.horizon,
            "agents": [a.name for a in self.agents],
            "rounds": rounds,
            "consensus": final,
            "consensus_round1": rounds[0]["consensus"],
            "revision": deltas,
            "errors": errors,
        }
