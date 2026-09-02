import json

import pytest

from agents.town_crier import TownCrierAgent
from data.digest_builder import HeuristicDigestBuilder
from data.news_feed import NewsItem

UNIVERSE = ["GLD", "TLT", "QQQ"]


def items():
    return [
        NewsItem("n1", "2016-06-24T19:00:00Z", "wire", "Gold surges as investors seek safe havens",
                 "bullion rallies", ["GLD"]),
        NewsItem("n2", "2016-06-24T19:05:00Z", "wire", "Treasury yields plunge on growth fears",
                 "duration bid", ["TLT"]),
        NewsItem("n3", "2016-06-24T19:10:00Z", "wire", "Gold miners extend gains", "", ["GLD"]),
    ]


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat(self, model, messages, temperature=0.0):
        self.calls += 1
        return json.dumps(self.payload)


class BrokenClient:
    def chat(self, **kwargs):
        raise RuntimeError("model down")


def crier(client=None, use_llm=False):
    return TownCrierAgent(HeuristicDigestBuilder(), client=client, model="m", use_llm=use_llm)


def test_heuristic_segment_reports_counts_stocks_and_questions():
    seg = crier().summarize_segment(items(), "T", UNIVERSE)
    assert seg.source == "heuristic"
    assert seg.n_items == 3
    assert seg.stocks[0] == "GLD"                     # mentioned twice
    assert seg.rag_questions["news"] and seg.rag_questions["insights"]
    assert "3 item(s)" in seg.summary


def test_segment_carries_the_digest_agents_still_need():
    seg = crier().summarize_segment(items(), "T", UNIVERSE)
    assert len(seg.digest["news_digest"]) == 3
    assert seg.digest["news_digest"][0]["sentiment"] in ("BULLISH", "BEARISH", "NEUTRAL")


def test_llm_segment_is_used_when_available():
    client = FakeClient({"summary": "Risk-off: gold bid, yields down.",
                         "stocks": ["GLD", "TLT"],
                         "rag_questions": {"news": ["past safe haven rallies"],
                                           "insights": ["how did I trade gold spikes"]}})
    seg = crier(client, use_llm=True).summarize_segment(items(), "T", UNIVERSE)
    assert seg.source == "llm"
    assert seg.summary.startswith("Risk-off")
    assert seg.rag_questions["news"] == ["past safe haven rallies"]


def test_llm_failure_falls_back_to_the_heuristic_segment():
    seg = crier(BrokenClient(), use_llm=True).summarize_segment(items(), "T", UNIVERSE)
    assert seg.source == "heuristic"
    assert seg.n_items == 3


def test_llm_tickers_outside_the_universe_are_dropped():
    client = FakeClient({"summary": "s", "stocks": ["GLD", "DOGE"], "rag_questions": {}})
    seg = crier(client, use_llm=True).summarize_segment(items(), "T", UNIVERSE)
    assert "DOGE" not in seg.stocks


def test_empty_segment_does_not_call_the_model():
    client = FakeClient({"summary": "unused"})
    seg = crier(client, use_llm=True).summarize_segment([], "T", UNIVERSE)
    assert client.calls == 0
    assert seg.n_items == 0


def test_context_summary_lists_headlines_without_a_model():
    docs = [{"date": "2013-07-22", "text": "Gold up sharply", "stocks": ["GLD"]},
            {"date": "2011-08-01", "text": "Debt ceiling standoff", "stocks": ["TLT"]}]
    summary = crier().summarize_context(docs, "gold")
    assert "2013-07-22" in summary and "Gold up sharply" in summary


def test_context_summary_uses_the_model_when_available():
    client = FakeClient({"context": "In 2011 and 2013 similar spikes faded within a week."})
    assert "faded" in crier(client, use_llm=True).summarize_context(
        [{"date": "2013-07-22", "text": "Gold up sharply"}], "gold")


def test_context_summary_is_empty_when_nothing_was_retrieved():
    assert crier().summarize_context([]) == ""
