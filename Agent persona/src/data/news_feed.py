"""News sources.

Every feed answers the same question - "what came out in this window, for these
tickers?" - so the simulator does not care whether it is replaying 2016 from the
local archive or polling live headlines every 15 minutes.

  ArchiveNewsFeed   876k-headline local archive (2009-2020). Backtests.
  YahooRssNewsFeed  live, no API key required. The 15-minute loop.
  FinnhubNewsFeed   live, needs FINNHUB_API_KEY. Better coverage than RSS.
  FixtureNewsFeed   a JSON file. Demos and tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; agentic-stock-sim/1.0)"}


@dataclass
class NewsItem:
    news_id: str
    time: str                       # ISO-8601 UTC
    source: str
    headline: str
    summary: str = ""
    tickers: List[str] = field(default_factory=list)
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "news_id": self.news_id, "time": self.time, "source": self.source,
            "headline": self.headline, "summary": self.summary,
            "tickers_mentioned": self.tickers, "url": self.url,
        }


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_id(*parts: str) -> str:
    return "n_" + hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


class NewsFeed(ABC):
    name: str = "feed"

    @abstractmethod
    def fetch(self, at: datetime, lookback_minutes: int, limit: int = 20) -> List[NewsItem]:
        """News published in (at - lookback, at], newest first."""

    def describe(self) -> Dict[str, Any]:
        return {"feed": self.name}


class ArchiveNewsFeed(NewsFeed):
    """Replay the local headline archive as if it were breaking news."""

    name = "archive"

    def __init__(self, universe: Sequence[str], db_path: Optional[Path] = None):
        root = Path(__file__).resolve().parents[3]
        self.db_path = Path(db_path or root / "dataset" / "headline_store.sqlite")
        self.universe = [t.upper() for t in universe]
        self._local = threading.local()

    def _conn(self) -> Optional[sqlite3.Connection]:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if not self.db_path.exists():
                return None
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self._local.conn = conn
        return conn

    def available(self) -> bool:
        return self.db_path.exists()

    def date_range(self) -> Dict[str, str]:
        conn = self._conn()
        if not conn:
            return {}
        lo, hi = conn.execute("SELECT MIN(date), MAX(date) FROM headlines").fetchone()
        return {"start": (lo or "")[:10], "end": (hi or "")[:10]}

    def fetch(self, at: datetime, lookback_minutes: int, limit: int = 20) -> List[NewsItem]:
        conn = self._conn()
        if not conn:
            return []
        start = at - timedelta(minutes=lookback_minutes)
        rows = conn.execute(
            "SELECT doc_id, title, date, stock FROM headlines WHERE date > ? AND date <= ? "
            "ORDER BY date DESC",
            (start.strftime("%Y-%m-%d %H:%M:%S+00:00"), at.strftime("%Y-%m-%d %H:%M:%S+00:00")),
        ).fetchall()

        wanted = set(self.universe)
        items: List[NewsItem] = []
        for doc_id, title, date, stock in rows:
            tickers = [s.strip().upper() for s in (stock or "").split(",") if s.strip()]
            hit = [t for t in tickers if t in wanted]
            if wanted and not hit:
                continue
            items.append(NewsItem(
                news_id=doc_id, time=str(date).replace(" ", "T")[:19] + "Z",
                source="archive", headline=title or "", tickers=tickers[:8],
            ))
            if len(items) >= limit:
                break
        return items

    def describe(self) -> Dict[str, Any]:
        return {"feed": self.name, "db": str(self.db_path), **self.date_range()}


class YahooRssNewsFeed(NewsFeed):
    """Live headlines per ticker from Yahoo Finance RSS. No API key."""

    name = "yahoo_rss"
    URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"

    def __init__(self, universe: Sequence[str], timeout_s: int = 20):
        self.universe = [t.upper() for t in universe]
        self.timeout_s = timeout_s
        self._seen: set = set()

    def _fetch_ticker(self, ticker: str) -> List[NewsItem]:
        try:
            resp = requests.get(self.URL, params={"s": ticker, "region": "US", "lang": "en-US"},
                                headers=HEADERS, timeout=self.timeout_s)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception:
            return []

        items = []
        for node in root.findall("./channel/item"):
            title = (node.findtext("title") or "").strip()
            if not title:
                continue
            pub = node.findtext("pubDate") or ""
            try:
                when = parsedate_to_datetime(pub).astimezone(timezone.utc)
            except (TypeError, ValueError):
                when = datetime.now(timezone.utc)
            items.append(NewsItem(
                news_id=node.findtext("guid") or _hash_id(ticker, title),
                time=_iso(when), source="yahoo_rss", headline=title,
                summary=(node.findtext("description") or "").strip()[:400],
                tickers=[ticker], url=(node.findtext("link") or "").strip(),
            ))
        return items

    def fetch(self, at: datetime, lookback_minutes: int, limit: int = 20) -> List[NewsItem]:
        cutoff = at - timedelta(minutes=lookback_minutes)
        merged: Dict[str, NewsItem] = {}
        for ticker in self.universe:
            for item in self._fetch_ticker(ticker):
                when = datetime.strptime(item.time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if when <= cutoff or when > at + timedelta(minutes=5):
                    continue
                if item.news_id in merged:      # same story tagged for several tickers
                    merged[item.news_id].tickers = sorted(set(merged[item.news_id].tickers + item.tickers))
                else:
                    merged[item.news_id] = item
        fresh = [i for i in merged.values() if i.news_id not in self._seen]
        self._seen.update(i.news_id for i in fresh)
        fresh.sort(key=lambda i: i.time, reverse=True)
        return fresh[:limit]


class FinnhubNewsFeed(NewsFeed):
    """Live company news from Finnhub. Needs FINNHUB_API_KEY."""

    name = "finnhub"
    URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, universe: Sequence[str], api_key: Optional[str] = None, timeout_s: int = 20):
        self.universe = [t.upper() for t in universe]
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self.timeout_s = timeout_s
        self._seen: set = set()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def fetch(self, at: datetime, lookback_minutes: int, limit: int = 20) -> List[NewsItem]:
        if not self.available:
            return []
        cutoff = at - timedelta(minutes=lookback_minutes)
        items: List[NewsItem] = []
        for ticker in self.universe:
            try:
                resp = requests.get(self.URL, timeout=self.timeout_s, params={
                    "symbol": ticker,
                    "from": (at - timedelta(days=2)).strftime("%Y-%m-%d"),
                    "to": at.strftime("%Y-%m-%d"),
                    "token": self.api_key,
                })
                resp.raise_for_status()
                payload = resp.json()
            except Exception:
                continue
            for row in payload if isinstance(payload, list) else []:
                when = datetime.fromtimestamp(int(row.get("datetime", 0)), tz=timezone.utc)
                if when <= cutoff or when > at + timedelta(minutes=5):
                    continue
                news_id = f"fh_{row.get('id')}"
                if news_id in self._seen:
                    continue
                self._seen.add(news_id)
                items.append(NewsItem(
                    news_id=news_id, time=_iso(when), source=str(row.get("source", "finnhub")),
                    headline=str(row.get("headline", "")), summary=str(row.get("summary", ""))[:400],
                    tickers=[ticker], url=str(row.get("url", "")),
                ))
        items.sort(key=lambda i: i.time, reverse=True)
        return items[:limit]


class FixtureNewsFeed(NewsFeed):
    """Serve a fixed digest file - deterministic demos and tests."""

    name = "fixture"

    def __init__(self, path: Path):
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.items = [
            NewsItem(
                news_id=str(i.get("news_id", _hash_id(str(i)))),
                time=str(i.get("time", payload.get("timestamp", ""))),
                source=str(i.get("source", "fixture")),
                headline=str(i.get("headline", "")),
                summary=str(i.get("summary", "")),
                tickers=list(i.get("tickers_mentioned", []) or []),
            )
            for i in payload.get("news_digest", [])
        ]

    def fetch(self, at: datetime, lookback_minutes: int, limit: int = 20) -> List[NewsItem]:
        return self.items[:limit]


def build_feed(kind: str, universe: Sequence[str], fixture_path: Optional[Path] = None) -> NewsFeed:
    kind = (kind or "archive").lower()
    if kind == "archive":
        return ArchiveNewsFeed(universe)
    if kind in ("yahoo", "yahoo_rss", "live"):
        return YahooRssNewsFeed(universe)
    if kind == "finnhub":
        return FinnhubNewsFeed(universe)
    if kind == "fixture":
        if not fixture_path:
            raise ValueError("fixture feed needs a path")
        return FixtureNewsFeed(fixture_path)
    raise ValueError(f"Unknown feed: {kind}")
