"""Raw headlines -> the DigestItem contract the agents consume.

Two builders:
  HeuristicDigestBuilder - lexicon based, free and deterministic. Used for long
    backtests where one LLM call per cycle per item would be prohibitive.
  LLMDigestBuilder - one extra LLM call per cycle to tag the whole batch. Better
    labels, real cost.

The digest is deliberately a thin layer: it tags sentiment and macro themes and
gets out of the way. The reasoning is the agents' job.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Sequence

from core.schema import extract_json, normalize_confidence
from core.types import Digest, DigestItem
from data.news_feed import NewsItem

BULLISH_TERMS = {
    "beat": 2, "beats": 2, "tops": 2, "surge": 2, "surges": 2, "soar": 2, "soars": 2,
    "rally": 2, "rallies": 2, "jumps": 1, "gains": 1, "upgrade": 2, "upgrades": 2,
    "raises": 1, "raised": 1, "record": 1, "strong": 1, "growth": 1, "outperform": 2,
    "buy": 1, "bullish": 2, "beat estimates": 3, "higher": 1, "boost": 1, "boosts": 1,
    "recovery": 1, "rebound": 2, "rebounds": 2, "optimism": 1, "beats estimates": 3,
}
BEARISH_TERMS = {
    "miss": 2, "misses": 2, "plunge": 3, "plunges": 3, "slump": 2, "slumps": 2,
    "falls": 1, "fell": 1, "drop": 1, "drops": 1, "downgrade": 2, "downgrades": 2,
    "cuts": 1, "cut": 1, "warns": 2, "warning": 2, "weak": 2, "weaker": 2, "loss": 1,
    "losses": 1, "bearish": 2, "sell": 1, "underperform": 2, "slides": 2, "slide": 2,
    "recession": 3, "crash": 3, "selloff": 2, "sell-off": 2, "fears": 2, "concerns": 1,
    "lawsuit": 1, "probe": 1, "halt": 1, "bankruptcy": 3, "default": 2, "layoffs": 2,
}
NEGATIONS = ("not ", "no ", "isn't", "won't", "fails to ", "despite ")

MACRO_TAG_PATTERNS = {
    "CPI": r"\b(cpi|inflation|price index|deflation)\b",
    "RATES": r"\b(rate hike|rate cut|interest rates?|yields?|treasur\w+|bond)\b",
    "FED": r"\b(fed|fomc|central bank|powell|yellen|bernanke|ecb|boj)\b",
    "JOBS": r"\b(jobs?|payrolls?|unemployment|labor market|jobless)\b",
    "GROWTH": r"\b(gdp|growth|recession|slowdown|expansion|pmi|ism)\b",
    "EARNINGS": r"\b(earnings|eps|revenue|guidance|quarter\w*|results)\b",
    "OIL": r"\b(oil|crude|brent|wti|opec|gasoline|energy prices?)\b",
    "GOLD": r"\b(gold|bullion|precious metals?)\b",
    "FX": r"\b(dollar|euro|yen|currency|forex|fx)\b",
    "TECH": r"\b(chip|semiconductor|ai|cloud|software|iphone|data center)\b",
    "POLICY": r"\b(tariff|trade war|sanction|regulation|stimulus|election|brexit)\b",
    "RISK_OFF": r"\b(selloff|sell-off|panic|fear|volatility|vix|flight to safety)\b",
}


def _tag(text: str) -> List[str]:
    low = text.lower()
    return [tag for tag, pattern in MACRO_TAG_PATTERNS.items() if re.search(pattern, low)]


def _score_sentiment(text: str) -> tuple[str, float]:
    low = " " + text.lower() + " "
    score = 0
    for term, weight in BULLISH_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\b", low):
            score += -weight if any(n in low for n in NEGATIONS) else weight
    for term, weight in BEARISH_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\b", low):
            score -= weight
    if score >= 2:
        return "BULLISH", min(0.4 + 0.1 * score, 0.85)
    if score <= -2:
        return "BEARISH", min(0.4 + 0.1 * abs(score), 0.85)
    if score == 1:
        return "BULLISH", 0.45
    if score == -1:
        return "BEARISH", 0.45
    return "NEUTRAL", 0.35


class HeuristicDigestBuilder:
    name = "heuristic"

    def build(self, items: Sequence[NewsItem], timestamp: str,
              universe: Optional[Sequence[str]] = None) -> Digest:
        wanted = {t.upper() for t in (universe or [])}
        digest_items: List[DigestItem] = []
        for item in items:
            text = f"{item.headline}. {item.summary}".strip()
            sentiment, confidence = _score_sentiment(text)
            tickers = [t for t in item.tickers if not wanted or t.upper() in wanted]
            digest_items.append({
                "news_id": item.news_id, "time": item.time, "source": item.source,
                "headline": item.headline, "summary": item.summary[:300],
                "tickers_mentioned": tickers or item.tickers[:4],
                "macro_tags": _tag(text) or ["GENERAL"],
                "sentiment": sentiment, "confidence": round(confidence, 2),
            })
        return {"timestamp": timestamp, "news_digest": digest_items}


DIGEST_SYSTEM = "You label financial news. Output ONLY a JSON object."
DIGEST_PROMPT = """
For each item, decide its market sentiment and macro themes.

Return {"items": [{"news_id", "sentiment": "BULLISH"|"BEARISH"|"NEUTRAL",
"confidence": 0..1, "macro_tags": [UPPERCASE_TAG, ...],
"tickers_mentioned": [ticker, ...], "summary": one clause}]}

Sentiment is the effect on the mentioned instruments, not the tone of the writing.
Label NEUTRAL when the item is not market-moving; most headlines are.
Only use tickers from the provided universe. Keep one object per input item.
"""


class LLMDigestBuilder:
    """Tag a whole cycle's headlines in one call, with a heuristic fallback."""

    name = "llm"

    def __init__(self, client, model: str, fallback: Optional[HeuristicDigestBuilder] = None):
        self.client = client
        self.model = model
        self.fallback = fallback or HeuristicDigestBuilder()

    def build(self, items: Sequence[NewsItem], timestamp: str,
              universe: Optional[Sequence[str]] = None) -> Digest:
        if not items:
            return {"timestamp": timestamp, "news_digest": []}

        payload = {
            "universe": list(universe or []),
            "items": [{"news_id": i.news_id, "headline": i.headline,
                       "summary": i.summary[:200], "tickers": i.tickers} for i in items],
        }
        try:
            content = self.client.chat(model=self.model, messages=[
                {"role": "system", "content": DIGEST_SYSTEM},
                {"role": "developer", "content": DIGEST_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ], temperature=0.0)
            labelled = {str(r.get("news_id")): r for r in extract_json(content).get("items", [])}
        except Exception:
            return self.fallback.build(items, timestamp, universe)

        base = self.fallback.build(items, timestamp, universe)
        for entry in base["news_digest"]:
            row = labelled.get(entry["news_id"])
            if not row:
                continue
            sentiment = str(row.get("sentiment", "")).upper()
            if sentiment in ("BULLISH", "BEARISH", "NEUTRAL"):
                entry["sentiment"] = sentiment
            entry["confidence"] = round(normalize_confidence(row.get("confidence"), entry["confidence"]), 2)
            tags = [str(t).upper() for t in (row.get("macro_tags") or []) if str(t).strip()]
            if tags:
                entry["macro_tags"] = tags[:6]
            if row.get("summary"):
                entry["summary"] = str(row["summary"])[:300]
        return base


def build_digest_builder(kind: str, client=None, model: str = "") -> Any:
    if kind == "llm":
        if client is None:
            raise ValueError("LLM digest builder needs a chat client")
        return LLMDigestBuilder(client, model)
    return HeuristicDigestBuilder()
