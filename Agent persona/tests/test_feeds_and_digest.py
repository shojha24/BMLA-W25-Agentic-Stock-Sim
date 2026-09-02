import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from data.digest_builder import HeuristicDigestBuilder, LLMDigestBuilder
from data.news_feed import ArchiveNewsFeed, FixtureNewsFeed, NewsItem, build_feed

ARCHIVE_DB = Path(__file__).resolve().parents[2] / "dataset" / "headline_store.sqlite"
needs_archive = pytest.mark.skipif(not ARCHIVE_DB.exists(),
                                   reason="run rag_prep/build_headline_store.py")


def items():
    return [
        NewsItem("n1", "2016-06-24T19:00:00Z", "wire", "Gold surges as investors seek safe havens",
                 "Bullion rallies after the vote.", ["GLD"]),
        NewsItem("n2", "2016-06-24T19:10:00Z", "wire", "Bank shares plunge on recession fears",
                 "Lenders slump amid warnings.", ["XLF"]),
        NewsItem("n3", "2016-06-24T19:20:00Z", "wire", "Central banks to meet next week", "", ["TLT"]),
    ]


def test_heuristic_sentiment_reads_direction_not_tone():
    digest = HeuristicDigestBuilder().build(items(), "2016-06-24T21:00:00Z", ["GLD", "TLT", "XLF"])
    by_id = {d["news_id"]: d for d in digest["news_digest"]}
    assert by_id["n1"]["sentiment"] == "BULLISH"
    assert by_id["n2"]["sentiment"] == "BEARISH"
    assert by_id["n3"]["sentiment"] == "NEUTRAL"
    assert "GOLD" in by_id["n1"]["macro_tags"]


def test_digest_matches_the_agent_contract():
    digest = HeuristicDigestBuilder().build(items(), "T", ["GLD"])
    for entry in digest["news_digest"]:
        assert set(entry) == {"news_id", "time", "source", "headline", "summary",
                              "tickers_mentioned", "macro_tags", "sentiment", "confidence"}
        assert 0.0 <= entry["confidence"] <= 1.0


def test_empty_news_produces_an_empty_digest():
    assert HeuristicDigestBuilder().build([], "T", ["GLD"])["news_digest"] == []


class _BadClient:
    def chat(self, **kwargs):
        raise RuntimeError("model down")


class _RelabelClient:
    def chat(self, model, messages, temperature=0.0):
        return json.dumps({"items": [{"news_id": "n3", "sentiment": "BEARISH",
                                      "confidence": 0.9, "macro_tags": ["FED"]}]})


def test_llm_digest_falls_back_to_heuristic_when_the_model_fails():
    digest = LLMDigestBuilder(_BadClient(), "m").build(items(), "T", ["GLD"])
    assert len(digest["news_digest"]) == 3


def test_llm_labels_override_the_heuristic_ones():
    digest = LLMDigestBuilder(_RelabelClient(), "m").build(items(), "T", ["GLD", "TLT", "XLF"])
    by_id = {d["news_id"]: d for d in digest["news_digest"]}
    assert by_id["n3"]["sentiment"] == "BEARISH"
    assert by_id["n3"]["macro_tags"] == ["FED"]
    assert by_id["n1"]["sentiment"] == "BULLISH"     # untouched items keep heuristic labels


def test_fixture_feed_round_trips_a_digest_file(tmp_path):
    path = tmp_path / "digest.json"
    path.write_text(json.dumps({"timestamp": "T", "news_digest": [
        {"news_id": "a", "headline": "hello", "summary": "", "tickers_mentioned": ["SPY"]}]}))
    got = FixtureNewsFeed(path).fetch(datetime.now(timezone.utc), 15)
    assert got[0].headline == "hello"


@needs_archive
def test_archive_feed_only_returns_news_inside_the_window():
    feed = ArchiveNewsFeed(["QQQ", "GLD", "TLT"])
    at = datetime(2016, 6, 24, 21, 0, tzinfo=timezone.utc)
    got = feed.fetch(at, lookback_minutes=60 * 24, limit=10)
    assert got
    for item in got:
        assert "2016-06-23" <= item.time[:10] <= "2016-06-24"
        assert set(item.tickers) & {"QQQ", "GLD", "TLT"}


@needs_archive
def test_archive_feed_is_empty_outside_its_coverage():
    feed = ArchiveNewsFeed(["QQQ"])
    assert feed.fetch(datetime(2030, 1, 1, tzinfo=timezone.utc), 60) == []


def test_feed_factory_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        build_feed("bloomberg", ["SPY"])
