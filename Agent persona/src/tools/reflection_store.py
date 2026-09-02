"""Vector Store 2: what each agent learned, kept private to that agent.

The whiteboard marks this store Private, and that is enforced here rather than
by convention: every read is scoped to one agent_id. An agent can retrieve its
own past lessons and nobody else's - otherwise the personas would converge into
one shared memory and the panel would stop disagreeing.

Same two tiers as the news index: SQLite FTS5 always, ChromaDB when a vector
store and an embedding key are both available.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS reflections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    agent_id   TEXT NOT NULL,
    day        TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    lesson     TEXT NOT NULL,
    what_worked TEXT DEFAULT '',
    what_failed TEXT DEFAULT '',
    tags       TEXT DEFAULT '',
    tickers    TEXT DEFAULT '',
    pnl_usd    REAL DEFAULT 0.0,
    n_trades   INTEGER DEFAULT 0,
    source     TEXT DEFAULT 'heuristic',
    payload    TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_reflections_agent ON reflections(agent_id, day);
CREATE VIRTUAL TABLE IF NOT EXISTS reflections_fts USING fts5(
    ref_id UNINDEXED, agent_id UNINDEXED, body, tokenize='porter'
);
"""

from tools.news_index import fts_query


class ReflectionStore:
    def __init__(self, db_path: Path | str, chroma_writer: Optional["ChromaReflectionWriter"] = None):
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

    def add(self, run_id: str, agent_id: str, day: str, timestamp: str,
            reflection: Dict[str, Any]) -> int:
        lesson = str(reflection.get("lesson") or "").strip()
        if not lesson:
            return 0

        worked = " | ".join(str(x) for x in (reflection.get("what_worked") or []))
        failed = " | ".join(str(x) for x in (reflection.get("what_failed") or []))
        tags = ",".join(str(t).upper() for t in (reflection.get("tags") or []))
        tickers = ",".join(str(t).upper() for t in (reflection.get("tickers") or []))

        conn = self._conn()
        cur = conn.execute(
            "INSERT INTO reflections (run_id, agent_id, day, timestamp, lesson, what_worked, "
            "what_failed, tags, tickers, pnl_usd, n_trades, source, payload) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, agent_id, day, timestamp, lesson, worked, failed, tags, tickers,
             float(reflection.get("pnl_usd", 0.0) or 0.0), int(reflection.get("n_trades", 0) or 0),
             str(reflection.get("source", "heuristic")), json.dumps(reflection)))
        ref_id = int(cur.lastrowid)
        conn.execute("INSERT INTO reflections_fts (ref_id, agent_id, body) VALUES (?,?,?)",
                     (ref_id, agent_id, f"{lesson} {worked} {failed} {tags} {tickers}"))
        conn.commit()

        if self.chroma_writer is not None:
            self.chroma_writer.add(ref_id, run_id, agent_id, day, timestamp,
                                   f"{lesson} {worked} {failed}")
        return ref_id

    # ---------------- read (always agent-scoped) ----------------

    def _rows(self, cursor) -> List[Dict[str, Any]]:
        out = []
        for row in cursor.fetchall():
            item = dict(row)
            item["tags"] = [t for t in (item.get("tags") or "").split(",") if t]
            item["tickers"] = [t for t in (item.get("tickers") or "").split(",") if t]
            item.pop("payload", None)
            out.append(item)
        return out

    def search(self, agent_id: str, query: str, top_k: int = 3,
               before_day: Optional[str] = None, run_id: Optional[str] = None
               ) -> List[Dict[str, Any]]:
        """Retrieve this agent's own past lessons. Never another agent's."""
        match = fts_query(query)
        if not match or not agent_id:
            return []
        sql = ("SELECT r.* FROM reflections_fts f JOIN reflections r ON r.id = f.ref_id "
               "WHERE reflections_fts MATCH ? AND r.agent_id = ?")
        params: List[Any] = [match, agent_id]
        if before_day:
            sql += " AND r.day < ?"
            params.append(before_day)
        if run_id:
            sql += " AND r.run_id = ?"
            params.append(run_id)
        sql += " ORDER BY f.rank LIMIT ?"
        params.append(top_k)
        try:
            return self._rows(self._conn().execute(sql, params))
        except sqlite3.OperationalError:
            return []

    def latest(self, agent_id: str, limit: int = 3, before_day: Optional[str] = None,
               run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM reflections WHERE agent_id = ?"
        params: List[Any] = [agent_id]
        if before_day:
            sql += " AND day < ?"
            params.append(before_day)
        if run_id:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return self._rows(self._conn().execute(sql, params))

    def count(self, agent_id: Optional[str] = None, run_id: Optional[str] = None) -> int:
        """Rows in the store. The file outlives a single run, so scope by run_id
        whenever you are reporting on one."""
        sql, params = "SELECT COUNT(*) FROM reflections", []
        clauses = []
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return int(self._conn().execute(sql, params).fetchone()[0])


class ChromaReflectionWriter:
    """Optional dense tier for reflections, in its own Chroma collection.

    Metadata carries agent_id and timestamp, exactly as the whiteboard specifies,
    so the privacy filter survives into the vector store.
    """

    def __init__(self, vector_store_path: Path | str, collection_name: str = "reflections",
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

    def add(self, ref_id: int, run_id: str, agent_id: str, day: str, timestamp: str,
            text: str) -> bool:
        if not self.available or not text.strip():
            return False
        try:
            from google.genai import types
            resp = self._client.models.embed_content(
                model=self.model, contents=[text],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
            self._collection.upsert(
                ids=[f"{run_id}:{agent_id}:{ref_id}"], documents=[text],
                embeddings=[list(resp.embeddings[0].values)],
                metadatas=[{"agent_id": agent_id, "run_id": run_id, "day": day,
                            "timestamp": timestamp}])
            return True
        except Exception as exc:
            self.error = f"reflection upsert failed: {exc}"
            return False

    def search(self, agent_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        try:
            from google.genai import types
            resp = self._client.models.embed_content(
                model=self.model, contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
            out = self._collection.query(
                query_embeddings=[list(resp.embeddings[0].values)], n_results=top_k,
                where={"agent_id": agent_id})        # privacy travels with the query
            return [{"lesson": doc, **(meta or {})}
                    for doc, meta in zip(out.get("documents", [[]])[0],
                                         out.get("metadatas", [[]])[0])]
        except Exception as exc:
            self.error = f"reflection query failed: {exc}"
            return []
