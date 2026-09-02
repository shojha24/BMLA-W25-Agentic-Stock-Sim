import pytest

from data.news_feed import NewsItem
from tools.news_index import LiveNewsIndex, fts_query


@pytest.fixture
def index(tmp_path):
    idx = LiveNewsIndex(tmp_path / "live.sqlite")
    idx.add([
        NewsItem("n1", "2016-06-24T19:00:00Z", "wire", "Gold surges as investors seek safe havens",
                 "bullion rallies after the vote", ["GLD"]),
        NewsItem("n2", "2016-06-25T19:00:00Z", "wire", "Nasdaq slides on rate fears",
                 "tech selloff deepens", ["QQQ"]),
    ])
    return idx


def test_items_are_indexed_once(index):
    assert index.count() == 2
    assert index.add([NewsItem("n1", "2016-06-24T19:00:00Z", "wire", "Gold surges", "", ["GLD"])]) == 0
    assert index.count() == 2


def test_search_finds_recent_news_by_keyword(index):
    hits = index.search("gold safe haven demand")
    assert hits and hits[0]["doc_id"] == "n1"
    assert "Gold surges" in hits[0]["text"]
    assert hits[0]["source"] == "live_index"


def test_search_respects_the_point_in_time_cutoff(index):
    assert [h["doc_id"] for h in index.search("nasdaq rate fears", cutoff_date="2016-06-24")] == []
    assert [h["doc_id"] for h in index.search("nasdaq rate fears", cutoff_date="2016-06-25")] == ["n2"]


def test_search_can_be_restricted_to_tickers(index):
    assert [h["doc_id"] for h in index.search("gold nasdaq", tickers=["QQQ"])] == ["n2"]


def test_rows_carry_their_tickers(index):
    assert index.search("gold")[0]["stocks"] == ["GLD"]


def test_punctuation_in_a_query_does_not_break_fts(index):
    """A headline pasted straight in as a query used to be a syntax error."""
    assert index.search('Gold "surges" (safe-havens): what now?') is not None
    assert index.search("AT&T / Q3 -- results!") == []


def test_empty_or_stopword_only_queries_return_nothing(index):
    assert index.search("") == []
    assert index.search("!!! ???") == []


def test_items_without_a_headline_are_skipped(tmp_path):
    idx = LiveNewsIndex(tmp_path / "live.sqlite")
    assert idx.add([NewsItem("n9", "2016-06-24T19:00:00Z", "wire", "", "", [])]) == 0


def test_dict_items_are_accepted_as_well_as_news_items(tmp_path):
    idx = LiveNewsIndex(tmp_path / "live.sqlite")
    assert idx.add([{"news_id": "d1", "time": "2016-06-24T19:00:00Z",
                     "headline": "Oil slumps on supply glut", "tickers_mentioned": ["XLE"]}]) == 1
    assert idx.search("oil supply glut")[0]["doc_id"] == "d1"


def test_fts_query_quotes_terms_and_drops_noise():
    assert fts_query("gold (safe-haven) demand!") == '"gold" OR "safe" OR "haven" OR "demand"'
    assert fts_query("...") == ""
