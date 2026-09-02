"""Vector Store 1, live half: index the news as it arrives.

The archive index is frozen at 2020-06-11, so without this the agents can
retrieve history but never anything from the run they are in. Each cycle's
headlines are written here and become retrievable by the next cycle.

Two tiers, same as retrieval:
  * SQLite FTS5 - always on, no key, no embedding cost.
  * ChromaDB    - optional, keeps the live news in the same collection as the
                  archive so dense search covers both. Needs GOOGLE_API_KEY.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_news (
    news_id  TEXT PRIMARY KEY,
    date     TEXT NOT NULL,
    stock    TEXT DEFAULT '',
    source   TEXT DEFAULT '',
    headline TEXT NOT NULL,
    summary  TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_live_news_date ON live_news(date);
CREATE VIRTUAL TABLE IF NOT EXISTS live_news_fts USING fts5(
    news_id UNINDEXED, body, tokenize='porter'
);
"""

# FTS5 reserves these; a headline is not a query language.
_FTS_SPECIALS = re.compile(r'[^\w\s]')


def fts_query(text: str, max_terms: int = 24) -> str:
    """Turn free text into a safe OR-query."""
    terms = [t for t in _FTS_SPECIALS.sub(" ", text).split() if len(t) > 1]
    seen, out = set(), []
    for term in terms:
        low = term.lower()
        if low not in seen:
            seen.add(low)
            out.append(f'"{low}"')
        if len(out) >= max_terms:
            break
    return " OR ".join(out)


class LiveNewsIndex:
    def __init__(self, db_path: Path | str, chroma_writer: Optional["ChromaNewsWriter"] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_writer = chroma_writer
        self._local = threading.local()
        self._conn().executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ---------------- write ----------------

    def add(self, items: Sequence[Any]) -> int:
        """Index news items (anything with news_id/time/headline). Ignores repeats."""
        rows = []
        for item in items:
            get = item.get if isinstance(item, dict) else lambda k, d=None: getattr(item, k, d)
            news_id = str(get("news_id", "") or "")
            headline = str(get("headline", "") or "")
            if not news_id or not headline:
                continue
            tickers = get("tickers", None)
            if tickers is None:
                tickers = get("tickers_mentioned", []) or []
            rows.append((
                news_id,
                str(get("time", "") or "")[:19].replace("T", " "),
                ",".join(str(t).upper() for t in tickers),
                str(get("source", "") or ""),
                headline,
                str(get("summary", "") or "")[:500],
            ))
        if not rows:
            return 0

        conn = self._conn()
        before = self.count()
        conn.executemany(
            "INSERT OR IGNORE INTO live_news (news_id, date, stock, source, headline, summary) "
            "VALUES (?,?,?,?,?,?)", rows)
        # Keep FTS in step only for rows that were actually new.
        conn.executemany(
            "INSERT INTO live_news_fts (news_id, body) "
            "SELECT ?, ? WHERE NOT EXISTS (SELECT 1 FROM live_news_fts WHERE news_id = ?)",
            [(r[0], f"{r[4]} {r[5]} {r[2]}", r[0]) for r in rows])
        conn.commit()
        added = self.count() - before

        if self.chroma_writer is not None and added:
            self.chroma_writer.add(rows)
        return added

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM live_news").fetchone()[0])

    # ---------------- read ----------------

    def search(self, query: str, top_k: int = 8, cutoff_date: Optional[str] = None,
               tickers: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        match = fts_query(query)
        if not match:
            return []
        sql = ("SELECT n.news_id, n.date, n.stock, n.headline, n.summary, f.rank "
               "FROM live_news_fts f JOIN live_news n ON n.news_id = f.news_id "
               "WHERE live_news_fts MATCH ?")
        params: List[Any] = [match]
        if cutoff_date:
            sql += " AND substr(n.date, 1, 10) <= ?"
            params.append(cutoff_date[:10])
        sql += " ORDER BY f.rank LIMIT ?"
        params.append(max(top_k * 4, top_k))

        try:
            rows = self._conn().execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []

        wanted = {t.upper() for t in (tickers or [])}
        out: List[Dict[str, Any]] = []
        for row in rows:
            stocks = [s for s in (row["stock"] or "").split(",") if s]
            if wanted and stocks and not wanted.intersection(stocks):
                continue
            out.append({
                "doc_id": row["news_id"],
                "date": row["date"],
                "stocks": stocks,
                "text": (row["headline"] + (f". {row['summary']}" if row["summary"] else ""))[:400],
                "source": "live_index",
            })
            if len(out) >= top_k:
                break
        return out


class ChromaNewsWriter:
    """Write live headlines into the same Chroma collection as the archive.

    Only usable when the vector store and an embedding key are both present;
    `available` is False otherwise and the SQLite tier carries the load alone.
    """

    def __init__(self, vector_store_path: Path | str, collection_name: str = "headlines",
                 api_key: Optional[str] = None, model: str = "models/gemini-embedding-001"):
        self.vector_store_path = Path(vector_store_path)
        self.collection_name = collection_name
        self.model = model
        self._collection = None
        self._client = None
        self.error = ""
        self._load(api_key)

    def _load(self, api_key: Optional[str]) -> None:
        import os
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.vector_store_path.exists():
            self.error = f"vector store not found at {self.vector_store_path}"
            return
        if not key:
            self.error = "GOOGLE_API_KEY not set"
            return
        try:
            import chromadb
            from google import genai
            self._collection = chromadb.PersistentClient(
                path=str(self.vector_store_path)).get_or_create_collection(name=self.collection_name)
            self._client = genai.Client(api_key=key)
        except Exception as exc:
            self.error = f"chroma/genai unavailable: {exc}"

    @property
    def available(self) -> bool:
        return self._collection is not None and self._client is not None

    def add(self, rows: Sequence[tuple]) -> int:
        if not self.available or not rows:
            return 0
        texts = [f"{r[4]} {r[5]}".strip() for r in rows]
        try:
            from google.genai import types
            resp = self._client.models.embed_content(
                model=self.model, contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
            vectors = [list(e.values) for e in resp.embeddings]
        except Exception as exc:
            self.error = f"embedding failed: {exc}"
            return 0
        try:
            self._collection.upsert(
                ids=[r[0] for r in rows], documents=texts, embeddings=vectors,
                metadatas=[{"date": r[1], "stock": r[2], "source": r[3] or "live"} for r in rows])
        except Exception as exc:
            self.error = f"chroma upsert failed: {exc}"
            return 0
        return len(rows)
