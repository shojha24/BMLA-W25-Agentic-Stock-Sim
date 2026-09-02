from pathlib import Path

import pytest

from tools.headline_store import HeadlineStore, generate_id
from tools.rag import RAGNewsTool, _parse_stocks

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "dataset" / "news_bm25_index"
STORE = ROOT / "dataset" / "headline_store.sqlite"

needs_index = pytest.mark.skipif(not INDEX.exists(), reason="BM25 index not present")
needs_store = pytest.mark.skipif(not STORE.exists(), reason="run rag_prep/build_headline_store.py")


def test_multi_stock_rows_are_parsed():
    assert _parse_stocks("FT,Y") == ["FT", "Y"]     # used to fail an exact-string match
    assert _parse_stocks("") == []


def test_doc_id_matches_the_ingest_recipe():
    assert generate_id("How Treasuries and ETFs Work", "2009-02-14 19:02:00+00:00", "NAV") == \
        "862374f761452531cdc16922363f287d"


@needs_store
def test_headline_store_returns_text():
    store = HeadlineStore(STORE)
    got = store.get_many(["862374f761452531cdc16922363f287d"])
    assert got["862374f761452531cdc16922363f287d"]["text"]


@needs_index
def test_status_reports_a_mode():
    status = RAGNewsTool().status()
    assert status["available"] is True
    assert status["mode"] in {"hybrid_dense_bm25", "bm25_local_text", "bm25_metadata_only"}


@needs_index
def test_retrieval_respects_the_point_in_time_cutoff():
    rows, err = RAGNewsTool().retrieve("inflation CPI yields", top_k=8, cutoff_date="2015-01-01")
    assert err == ""
    assert rows
    assert all(r["date"][:10] <= "2015-01-01" for r in rows)


@needs_index
def test_ticker_filter_only_returns_matching_rows():
    tickers = ["XLE", "SPY"]
    rows, err = RAGNewsTool().retrieve("oil supply", top_k=6, stock_filter=tickers)
    assert err == ""
    assert all(set(r["stocks"]) & set(tickers) for r in rows if r["stocks"])


@needs_index
@needs_store
def test_barbell_returns_both_recent_and_historical():
    rows, _ = RAGNewsTool().retrieve("inflation rate hike", top_k=8, cutoff_date="2019-01-01")
    assert {r["label"] for r in rows} == {"RECENT", "HISTORICAL"}
    assert any(r["text"] for r in rows)


@needs_index
def test_empty_query_is_reported_not_crashed():
    rows, err = RAGNewsTool().retrieve("   ")
    assert rows == [] and err
