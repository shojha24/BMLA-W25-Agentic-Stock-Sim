"""The Town Crier: reads the raw segment so the traders do not have to.

Two jobs from the whiteboard:
  1. Summarize the docs in the time span, and say which stocks they touch.
  2. Emit the questions retrieval should ask ("RAG-Qs") - one set about news,
     one about the agents' own past performance.

Retrieval then runs once, centrally, and the Town Crier condenses what came
back into the "Historical Context" line of the brief. Before this, every agent
hand-rolled its own query string and read raw headlines.

Runs with an LLM when one is available and falls back to deterministic
heuristics otherwise, so backtests are not forced to pay for it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from core.schema import extract_json
from core.types import Digest

CRIER_SYSTEM = "You are a market news editor. Output ONLY a single JSON object."

CRIER_PROMPT = """
You are the Town Crier for a trading desk. You have the raw news from one time
segment. Produce the segment brief the traders will read.

Return {"summary": string, "stocks": [ticker, ...], "rag_questions":
{"news": [question, ...], "insights": [question, ...]}}

- "summary": 2-4 sentences. What happened, and what it plausibly means for the
  instruments in the universe. No hedging boilerplate, no restating headlines
  one by one. If the segment is thin, say so plainly.
- "stocks": the universe tickers this segment actually bears on, most affected first.
- "rag_questions.news": up to 4 retrieval queries for a historical news archive -
  what past episodes would help price this? Write them as search queries, not chat.
- "rag_questions.insights": up to 3 questions a trader should ask about their own
  past performance in situations like this.
"""

CONTEXT_PROMPT = """
Condense these historical headlines into the "Historical Context" line of a
trading brief: 2-4 sentences on what past episodes like this one did to markets.
Cite dates. If the headlines are not actually relevant to the query, say that
instead of forcing a narrative. Return {"context": string}.
"""


@dataclass
class SegmentBrief:
    timestamp: str
    summary: str
    stocks: List[str]
    rag_questions: Dict[str, List[str]]
    digest: Digest
    n_items: int
    source: str = "heuristic"          # which path produced the summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp, "summary": self.summary, "stocks": self.stocks,
            "rag_questions": self.rag_questions, "n_items": self.n_items, "source": self.source,
        }


class TownCrierAgent:
    name = "town_crier"

    def __init__(self, digest_builder, client=None, model: str = "", use_llm: bool = True,
                 max_context_docs: int = 8):
        self.digest_builder = digest_builder
        self.client = client
        self.model = model
        self.use_llm = use_llm and client is not None
        self.max_context_docs = max_context_docs

    # ---------------- segment ----------------

    def summarize_segment(self, items: Sequence[Any], timestamp: str,
                          universe: Sequence[str]) -> SegmentBrief:
        digest = self.digest_builder.build(items, timestamp, universe)
        entries = digest.get("news_digest", [])
        heuristic = self._heuristic_segment(entries, universe)

        if not self.use_llm or not entries:
            return SegmentBrief(timestamp=timestamp, digest=digest, n_items=len(entries),
                                **heuristic)

        payload = {
            "timestamp": timestamp,
            "universe": list(universe),
            "items": [{"news_id": e["news_id"], "headline": e["headline"],
                       "summary": e["summary"][:200], "tickers": e["tickers_mentioned"],
                       "tags": e["macro_tags"], "sentiment": e["sentiment"]} for e in entries],
        }
        try:
            content = self.client.chat(model=self.model, messages=[
                {"role": "system", "content": CRIER_SYSTEM},
                {"role": "developer", "content": CRIER_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ], temperature=0.0)
            raw = extract_json(content)
        except Exception:
            return SegmentBrief(timestamp=timestamp, digest=digest, n_items=len(entries),
                                **heuristic)

        summary = str(raw.get("summary") or "").strip() or heuristic["summary"]
        stocks = [str(t).upper() for t in (raw.get("stocks") or []) if str(t).strip()]
        stocks = [t for t in stocks if t in {u.upper() for u in universe}] or heuristic["stocks"]

        questions = raw.get("rag_questions") or {}
        news_qs = [str(q) for q in (questions.get("news") or []) if str(q).strip()][:4]
        insight_qs = [str(q) for q in (questions.get("insights") or []) if str(q).strip()][:3]

        return SegmentBrief(
            timestamp=timestamp, summary=summary[:1200], stocks=stocks[:12],
            rag_questions={"news": news_qs or heuristic["rag_questions"]["news"],
                           "insights": insight_qs or heuristic["rag_questions"]["insights"]},
            digest=digest, n_items=len(entries), source="llm",
        )

    @staticmethod
    def _heuristic_segment(entries: Sequence[Dict[str, Any]],
                           universe: Sequence[str]) -> Dict[str, Any]:
        """Deterministic fallback: frequency counts, no model."""
        wanted = {u.upper() for u in universe}
        ticker_hits: Dict[str, int] = {}
        tag_hits: Dict[str, int] = {}
        tone = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
        for e in entries:
            for t in e.get("tickers_mentioned", []) or []:
                key = str(t).upper()
                if key in wanted:
                    ticker_hits[key] = ticker_hits.get(key, 0) + 1
            for tag in e.get("macro_tags", []) or []:
                tag_hits[str(tag).upper()] = tag_hits.get(str(tag).upper(), 0) + 1
            tone[str(e.get("sentiment", "NEUTRAL")).upper()] = \
                tone.get(str(e.get("sentiment", "NEUTRAL")).upper(), 0) + 1

        stocks = [t for t, _ in sorted(ticker_hits.items(), key=lambda kv: -kv[1])][:12]
        tags = [t for t, _ in sorted(tag_hits.items(), key=lambda kv: -kv[1])][:6]
        headlines = [e.get("headline", "") for e in entries[:3] if e.get("headline")]

        summary = (
            f"{len(entries)} item(s) in this segment "
            f"({tone.get('BULLISH', 0)} bullish / {tone.get('BEARISH', 0)} bearish / "
            f"{tone.get('NEUTRAL', 0)} neutral). "
            f"Themes: {', '.join(tags) or 'none tagged'}. "
            f"Top headlines: {' | '.join(headlines) or 'none'}."
        )
        return {
            "summary": summary,
            "stocks": stocks or [u.upper() for u in universe][:6],
            "rag_questions": {
                "news": [q for q in [
                    " ".join(headlines[:2]),
                    f"{' '.join(tags[:4])} market reaction",
                    f"{' '.join(stocks[:4])} {' '.join(tags[:2])}",
                ] if q.strip()],
                "insights": [f"past trades in {' '.join(tags[:3]) or 'similar conditions'}"],
            },
            "source": "heuristic",
        }

    # ---------------- retrieved context ----------------

    def summarize_context(self, rows: Sequence[Dict[str, Any]], question: str = "") -> str:
        """Condense retrieved archive rows into the brief's Historical Context line."""
        rows = [r for r in rows if r.get("text")][: self.max_context_docs]
        if not rows:
            return ""
        fallback = " | ".join(f"[{r.get('date', '')[:10]}] {r['text'][:110]}" for r in rows[:5])
        if not self.use_llm:
            return fallback

        payload = {"query": question, "headlines": [
            {"date": r.get("date", "")[:10], "text": r["text"][:200],
             "stocks": r.get("stocks", [])} for r in rows]}
        try:
            content = self.client.chat(model=self.model, messages=[
                {"role": "system", "content": CRIER_SYSTEM},
                {"role": "developer", "content": CONTEXT_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ], temperature=0.0)
            return str(extract_json(content).get("context") or "").strip()[:1200] or fallback
        except Exception:
            return fallback
