"""One LLM agent implementation, three personalities.

A `PersonaSpec` carries everything that differs between personas (prompt,
retrieval strategy, instrument preferences); `LLMPersonaAgent` carries
everything that does not (RAG call, payload assembly, LLM call, validation,
round-2 revision).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from core.schema import extract_json, normalize_agent_output
from core.types import AgentOutput, Digest, State
from llm.base import ChatClient
from tools.rag import RAGNewsTool

SHARED_SYSTEM = (
    "You are a trading research agent in a multi-agent simulation. "
    "You must output ONLY a single valid JSON object: no markdown, no commentary."
)

OUTPUT_CONTRACT = """
Output a SINGLE JSON object with exactly these keys:
- "decision": one sentence stating your net stance.
- "market_view": {"risk_regime": "RISK_ON"|"RISK_OFF"|"NEUTRAL", "confidence": 0..1,
  "summary": string, "key_drivers": [news_id, ...]}
- "signals": <= 8 objects, each {"news_id", "macro_tags": [...], "sentiment":
  "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0..1, "interpretation": string}
- "forecasts": one object per ticker you have a view on, each
  {"ticker", "direction": "UP"|"DOWN"|"FLAT", "expected_return_bps": signed number,
   "horizon": "{horizon}", "confidence": 0..1, "rationale": string, "news_refs": [news_id|doc_id, ...]}
- "actions": the orders you are placing this cycle, each
  {"side": "BUY"|"SELL", "ticker", "qty": whole number of shares, "rationale",
   "news_refs": [...]}. Use [] to trade nothing.
- "checks": {"digest_items": int, "universe_prices_keys": int}

"forecasts" and "actions" are two different jobs. Do not let caution about one
suppress the other.

Hard constraints on "forecasts" (your view, scored every cycle):
- Emit one forecast for EVERY ticker in state.prices. Never invent a ticker.
- FLAT is a claim that the move will be smaller than +/-10 bps, not a way to say
  "I am unsure". Uncertainty belongs in "confidence". A thin digest means low
  confidence with a direction, not a wall of FLAT.
- expected_return_bps is the signed move over {horizon} in basis points (100 bps = 1%).
  Its sign must agree with "direction".
- confidence is calibrated: reserve > 0.8 for news that is unambiguous for that ticker.

Hard constraints on "actions" (real orders against your own book):
- You may only BUY with the cash in state.cash_usd, at the prices in state.prices.
  An order you cannot pay for is cut down or rejected by the venue.
- You may only SELL shares you actually hold in state.positions. No shorting.
- qty is a whole number of shares. No single order may exceed {max_order_pct}% of your equity.
- Trade when your own forecast is directional and you have the conviction for it;
  an empty list is acceptable when nothing clears your bar, but standing aside every
  cycle means the book never expresses your views.
- Actions must agree with your forecasts: never buy what you forecast DOWN.
- rag_context is historical background retrieved from a 2009-2020 news archive.
  It is supporting evidence only; the current digest dominates.
Output JSON ONLY.
"""

REVISION_BLOCK = """
This is round 2 of a roundtable. You are shown your own round-1 output and the
other agents' forecasts and reasoning.

- Update a forecast only when a peer supplies a fact or mechanism you had missed;
  disagreement alone is not evidence. Holding your view is a valid outcome.
- Peers see the same digest, so agreeing with them adds no information unless
  their reasoning changed yours.
- Add a "revision_note" string to market_view saying what you changed and why
  (or that you held).
- Your "actions" are re-submitted from scratch each round: round 1's orders were not
  sent to the venue. Emit the orders you actually want executed now.
Emit the same JSON schema as round 1.
"""


@dataclass
class PersonaSpec:
    key: str
    name: str
    persona: str
    mandate: str                                   # persona-specific developer prompt body
    rag_query_prefix: str = ""
    preferred_tickers: List[str] = field(default_factory=list)
    default_horizon: str = "1d"
    temperature: float = 0.0
    rag_top_k: int = 8


class LLMPersonaAgent(BaseAgent):
    def __init__(
        self,
        spec: PersonaSpec,
        client: ChatClient,
        model: str = "minimax/minimax-m3:free",
        rag_tool: Optional[RAGNewsTool] = None,
        use_rag: bool = True,
        max_order_pct_equity: float = 0.35,
    ):
        self.spec = spec
        self.max_order_pct_equity = max_order_pct_equity
        self.name = spec.name
        self.persona = spec.persona
        self.client = client
        self.model = model
        self.use_rag = use_rag
        self.rag_tool = rag_tool if rag_tool is not None else RAGNewsTool()

    # ---------------- retrieval ----------------

    def build_rag_query(self, digest: Digest, state: State) -> str:
        items = (digest.get("news_digest") or [])[:6]
        headlines = [i.get("headline", "") for i in items if i.get("headline")]
        tags: List[str] = []
        for i in items:
            tags.extend(i.get("macro_tags", []) or [])
        unique_tags = list(dict.fromkeys(tags))[:10]
        tickers = list((state.get("prices") or {}).keys())[:12]
        parts = [
            self.spec.rag_query_prefix,
            " ".join(headlines),
            " ".join(unique_tags),
            " ".join(tickers),
        ]
        return " | ".join(p for p in parts if p.strip())

    def _retrieve(self, digest: Digest, state: State) -> Dict[str, Any]:
        if not self.use_rag:
            return {"context": [], "meta": {"status": "disabled", "query": "", "error": ""}}
        query = self.build_rag_query(digest, state)
        if not query:
            return {"context": [], "meta": {"status": "empty_query", "query": "", "error": ""}}
        cutoff = str(digest.get("timestamp") or "")[:10] or None
        rows, err = self.rag_tool.retrieve(
            query=query,
            top_k=self.spec.rag_top_k,
            stock_filter=list((state.get("prices") or {}).keys()),
            cutoff_date=cutoff,
        )
        return {
            "context": rows,
            "meta": {
                "status": "ok" if rows else ("error" if err else "empty"),
                "query": query,
                "error": err,
                "mode": self.rag_tool.status().get("mode", "unknown"),
                "ticker_filtered": bool(rows and rows[0].get("ticker_filtered")),
            },
        }

    # ---------------- prompting ----------------

    def _developer_prompt(self, revision: bool) -> str:
        contract = (OUTPUT_CONTRACT
                    .replace("{horizon}", self.spec.default_horizon)
                    .replace("{max_order_pct}", str(int(self.max_order_pct_equity * 100))))
        blocks = [
            f"Persona: {self.spec.persona}",
            self.spec.mandate.strip(),
            contract,
        ]
        if revision:
            blocks.append(REVISION_BLOCK)
        return "\n\n".join(blocks)

    @staticmethod
    def _peer_brief(peer_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        brief = []
        for peer in peer_context:
            brief.append({
                "agent_name": peer.get("agent_name"),
                "persona": peer.get("persona"),
                "risk_regime": (peer.get("market_view") or {}).get("risk_regime"),
                "summary": (peer.get("market_view") or {}).get("summary"),
                "forecasts": [
                    {k: f.get(k) for k in
                     ("ticker", "direction", "expected_return_bps", "confidence", "rationale")}
                    for f in (peer.get("forecasts") or [])
                ],
            })
        return brief

    def run(
        self,
        digest: Digest,
        state: State,
        peer_context: Optional[List[Dict[str, Any]]] = None,
        prior_output: Optional[AgentOutput] = None,
    ) -> AgentOutput:
        rag = self._retrieve(digest, state)
        timestamp = str(digest.get("timestamp") or "")

        payload: Dict[str, Any] = {
            "timestamp": timestamp,
            "horizon": self.spec.default_horizon,
            "news_digest": digest.get("news_digest", []),
            "state": state,
            "preferred_tickers": self.spec.preferred_tickers,
            "trading_rules": {
                "cash_available_usd": (state or {}).get("cash_usd", 0.0),
                "max_order_pct_equity": self.max_order_pct_equity,
                "shorting_allowed": False,
                "whole_shares_only": True,
            },
            "rag_context": rag["context"],
            "rag_meta": rag["meta"],
        }

        revision = bool(peer_context)
        if revision:
            peers = self._peer_brief(peer_context or [])
            payload["peers"] = peers
            payload["peer_forecasts"] = [f for p in peers for f in p["forecasts"]]
            if prior_output:
                payload["your_round1"] = {
                    "market_view": prior_output.get("market_view"),
                    "forecasts": prior_output.get("forecasts"),
                }

        messages = [
            {"role": "system", "content": SHARED_SYSTEM},
            {"role": "developer", "content": self._developer_prompt(revision)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        content = self.client.chat(
            model=self.model, messages=messages, temperature=self.spec.temperature
        )
        raw = extract_json(content)
        out = normalize_agent_output(
            raw,
            agent_name=self.name,
            persona=self.persona,
            timestamp=timestamp,
            universe=list((state.get("prices") or {}).keys()),
            default_horizon=self.spec.default_horizon,
        )
        out["checks"] = {
            **out.get("checks", {}),
            "rag_status": rag["meta"].get("status"),
            "rag_docs": len(rag["context"]),
            "round": 2 if revision else 1,
        }
        return out
