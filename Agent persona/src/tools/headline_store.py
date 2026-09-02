"""Local doc_id -> headline text lookup.

The BM25 corpus only stores {doc_id, date, stock}; the headline text lives in
the ChromaDB vector store, which is distributed out-of-band (Google Drive) and
is absent from a fresh clone. But doc_id is reproducible from the source CSVs
(md5 of "date|stock|title"), so the text can be rebuilt locally into SQLite.

That makes RAG return real headlines with no Chroma, no embeddings and no API
key. Build it with:  python rag_prep/build_headline_store.py
"""
from __future__ import annotations

import csv
import hashlib
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_DB_NAME = "headline_store.sqlite"


def generate_id(content: str, date: str, stock: str) -> str:
    """Must match rag_prep/ingest.py exactly or the ids will not line up."""
    return hashlib.md5(f"{date}|{stock}|{content}".encode()).hexdigest()


class HeadlineStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        # One connection per thread: the roundtable queries this concurrently,
        # and a single sqlite3 connection is not safe to share across threads.
        self._local = threading.local()

    @property
    def available(self) -> bool:
        return self.db_path.exists()

    def _connect(self) -> Optional[sqlite3.Connection]:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if not self.available:
            return None
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self._local.conn = conn
        return conn

    def get_many(self, doc_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
        ids = [d for d in doc_ids if d]
        conn = self._connect()
        if not conn or not ids:
            return {}
        out: Dict[str, Dict[str, str]] = {}
        # SQLite caps variables per statement; chunk to stay well under it.
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT doc_id, title, date, stock FROM headlines WHERE doc_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for doc_id, title, date, stock in rows:
                out[doc_id] = {"text": title or "", "date": date or "", "stock": stock or ""}
        return out

    def count(self) -> int:
        conn = self._connect()
        if not conn:
            return 0
        return int(conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0])

    # ---------------- build ----------------

    @classmethod
    def build(
        cls,
        csv_paths: List[Path],
        db_path: Path,
        content_col: str = "title",
        batch: int = 20000,
        verbose: bool = True,
    ) -> int:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute(
            "CREATE TABLE headlines (doc_id TEXT PRIMARY KEY, title TEXT, date TEXT, stock TEXT)"
        )

        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        total = 0
        rows: List[tuple] = []
        for path in csv_paths:
            if not path.exists():
                if verbose:
                    print(f"  skip (missing): {path}")
                continue
            if verbose:
                print(f"  reading {path.name} ...")
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    title = (row.get(content_col) or "").strip()
                    if not title:
                        continue
                    date = row.get("date", "") or ""
                    stock = row.get("stock", "") or ""
                    rows.append((generate_id(title, date, stock), title, date, stock))
                    if len(rows) >= batch:
                        conn.executemany("INSERT OR IGNORE INTO headlines VALUES (?,?,?,?)", rows)
                        total += len(rows)
                        rows = []
                        if verbose and total % 200000 == 0:
                            print(f"    {total:,} rows")
        if rows:
            conn.executemany("INSERT OR IGNORE INTO headlines VALUES (?,?,?,?)", rows)
            total += len(rows)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_headlines_date ON headlines(date)")
        conn.commit()
        stored = int(conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0])
        conn.close()
        if verbose:
            print(f"  wrote {stored:,} unique rows ({total:,} read) -> {db_path}")
        return stored
