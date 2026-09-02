"""Retrieval tool the agents call.

Ports the hybrid search that previously lived only in the root-level research
scripts (search_interleaved.py / search_w_filter_decay.py) into something the
agents can actually use:

  * BM25 lexical search over the 876k-headline archive
  * optional dense search over ChromaDB (Gemini embeddings), fused with RRF
  * point-in-time cutoff (never retrieve news published after the digest)
  * ticker filter that understands multi-stock rows ("FT,Y")
  * exponential recency decay, plus a "barbell" interleave that returns both
    the most recent and the most relevant historical matches
  * headline text from ChromaDB when present, else from the local SQLite store

It degrades in steps instead of failing: check `status()["mode"]` to see which
tier is live.
"""
from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.headline_store import DEFAULT_DB_NAME, HeadlineStore
from tools.news_index import LiveNewsIndex

DECAY_RATE = 0.995          # a story keeps ~16% of its weight after a year
RRF_K = 60                  # reciprocal-rank-fusion constant
OVERSAMPLE = 400            # fetch this many before filtering by date/ticker


def _parse_stocks(raw: Any) -> List[str]:
    if not raw:
        return []
    return [s.strip().upper() for s in str(raw).split(",") if s.strip()]


def _day(date_str: str) -> str:
    return str(date_str or "")[:10]


class RAGNewsTool:
    def __init__(
        self,
        bm25_index_path: Optional[Path] = None,
        vector_store_path: Optional[Path] = None,
        headline_db_path: Optional[Path] = None,
        collection_name: str = "headlines",
        decay_rate: float = DECAY_RATE,
        use_dense: bool = True,
        top_k_fallback: int = 8,
        live_index: Optional[LiveNewsIndex] = None,
    ):
        root = Path(__file__).resolve().parents[3]
        self.bm25_index_path = Path(bm25_index_path or root / "dataset" / "news_bm25_index")
        self.vector_store_path = Path(vector_store_path or root / "dataset" / "vector_store")
        self.headline_db_path = Path(headline_db_path or root / "dataset" / DEFAULT_DB_NAME)
        self.collection_name = collection_name
        self.decay_rate = decay_rate
        self.use_dense = use_dense
        self.top_k_fallback = top_k_fallback
        # News from the current run: the archive stops in 2020, this does not.
        self.live_index = live_index

        self._loaded = False
        self._available = False
        self._error = ""
        self._notes: List[str] = []
        self._bm25 = None
        self._stemmer = None
        self._collection = None
        self._genai_client = None
        self._store = HeadlineStore(self.headline_db_path)

    # ---------------- loading ----------------

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        try:
            import bm25s
            import Stemmer
        except Exception as exc:
            self._error = (
                f"BM25 dependencies missing ({exc}). Install with: pip install -r requirements.txt"
            )
            return

        if not self.bm25_index_path.exists():
            self._error = (
                f"BM25 index not found at {self.bm25_index_path}. "
                "It ships in the repo under dataset/news_bm25_index."
            )
            return

        try:
            # mmap=True mis-parses this corpus format on bm25s>=0.2, so load it in memory (~4s once).
            self._bm25 = bm25s.BM25.load(str(self.bm25_index_path), load_corpus=True)
            self._stemmer = Stemmer.Stemmer("english")
        except Exception as exc:
            self._error = f"Failed to load BM25 index: {exc}"
            return

        self._available = True

        if not self._store.available:
            self._notes.append(
                f"headline store missing at {self.headline_db_path} "
                "(run: python rag_prep/build_headline_store.py) - results will lack text "
                "unless ChromaDB is present"
            )

        if self.use_dense:
            self._load_dense()

    def _load_dense(self) -> None:
        if not self.vector_store_path.exists():
            self._notes.append(
                f"vector store missing at {self.vector_store_path} - dense search disabled, BM25 only"
            )
            return
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.vector_store_path))
            self._collection = client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:
            self._notes.append(f"ChromaDB unavailable ({exc}) - dense search disabled")
            return

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self._notes.append("GOOGLE_API_KEY not set - dense search disabled, using BM25 ranking only")
            return
        try:
            from google import genai
            self._genai_client = genai.Client(api_key=api_key)
        except Exception as exc:
            self._notes.append(f"google-genai unavailable ({exc}) - dense search disabled")

    @property
    def dense_enabled(self) -> bool:
        return self._collection is not None and self._genai_client is not None

    # ---------------- pieces ----------------

    def _embed(self, text: str) -> List[float]:
        if not self._genai_client:
            return []
        try:
            from google.genai import types
            resp = self._genai_client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return list(resp.embeddings[0].values)
        except Exception as exc:
            self._notes.append(f"query embedding failed ({exc}) - falling back to BM25 only")
            return []

    @staticmethod
    def _keep(doc_date: str, doc_stocks: List[str], cutoff_date: Optional[str],
              stock_set: Optional[set]) -> bool:
        if cutoff_date:
            # Point-in-time discipline: a backtest may never see tomorrow's news.
            if not doc_date or _day(doc_date) > _day(cutoff_date):
                return False
        if stock_set and doc_stocks and not stock_set.intersection(doc_stocks):
            return False
        return True

    def _bm25_search(self, query: str, limit: int, cutoff_date: Optional[str],
                     stock_set: Optional[set]) -> List[Dict[str, Any]]:
        import bm25s  # loaded lazily in _lazy_load; re-imported here for tokenize()
        tokens = bm25s.tokenize(query, stemmer=self._stemmer, show_progress=False)
        num_docs = int((getattr(self._bm25, "scores", None) or {}).get("num_docs") or OVERSAMPLE)
        k = max(min(OVERSAMPLE, num_docs), min(limit, num_docs))
        results, _ = self._bm25.retrieve(tokens, k=k, show_progress=False)

        rows: List[Dict[str, Any]] = []
        for item in results[0]:
            if isinstance(item, dict):
                doc_id, date, stocks = item.get("doc_id"), item.get("date", ""), _parse_stocks(item.get("stock"))
            else:
                doc_id, date, stocks = str(item), "", []
            if not doc_id or not self._keep(date, stocks, cutoff_date, stock_set):
                continue
            rows.append({"doc_id": doc_id, "date": date, "stocks": stocks})
            if len(rows) >= limit:
                break
        return rows

    def _dense_search(self, query: str, limit: int, cutoff_date: Optional[str],
                      stock_list: Optional[List[str]]) -> List[Dict[str, Any]]:
        if not self.dense_enabled:
            return []
        emb = self._embed(query)
        if not emb:
            return []
        where = {"stock": {"$in": stock_list}} if stock_list else None
        try:
            resp = self._collection.query(query_embeddings=[emb], n_results=OVERSAMPLE, where=where)
        except Exception as exc:
            self._notes.append(f"Chroma query failed ({exc})")
            return []

        rows: List[Dict[str, Any]] = []
        for doc_id, meta in zip(resp.get("ids", [[]])[0], resp.get("metadatas", [[]])[0]):
            meta = meta or {}
            date = str(meta.get("date", ""))
            stocks = _parse_stocks(meta.get("stock"))
            if not self._keep(date, stocks, cutoff_date, None):
                continue
            rows.append({"doc_id": doc_id, "date": date, "stocks": stocks})
            if len(rows) >= limit:
                break
        return rows

    def _live_search(self, query: str, limit: int, cutoff_date: Optional[str],
                     stock_set: Optional[set]) -> List[Dict[str, Any]]:
        if self.live_index is None:
            return []
        try:
            return self.live_index.search(query, top_k=limit, cutoff_date=cutoff_date,
                                          tickers=stock_set)
        except Exception as exc:
            self._notes.append(f"live index search failed ({exc})")
            return []

    def _decay(self, score: float, doc_date: str, cutoff_date: Optional[str]) -> float:
        if not doc_date:
            return score
        try:
            doc_dt = datetime.strptime(_day(doc_date), "%Y-%m-%d")
            ref_dt = datetime.strptime(_day(cutoff_date), "%Y-%m-%d") if cutoff_date else datetime.now()
        except ValueError:
            return score
        days = max((ref_dt - doc_dt).days, 0)
        return score * math.pow(self.decay_rate, days)

    def _hydrate(self, rows: List[Dict[str, Any]]) -> str:
        """Attach headline text. Returns which source supplied it."""
        doc_ids = [r["doc_id"] for r in rows if not r.get("text")]
        if not doc_ids:
            return "live_index" if rows else "none"

        texts: Dict[str, str] = {}
        source = "none"
        if self._collection is not None:
            try:
                got = self._collection.get(ids=doc_ids)
                for i, doc_id in enumerate(got.get("ids", []) or []):
                    doc = (got.get("documents") or [])[i]
                    if doc:
                        texts[doc_id] = doc
                if texts:
                    source = "chromadb"
            except Exception:
                pass
        if len(texts) < len(doc_ids) and self._store.available:
            for doc_id, rec in self._store.get_many([d for d in doc_ids if d not in texts]).items():
                texts[doc_id] = rec["text"]
            source = "chromadb+sqlite" if source == "chromadb" else "sqlite"

        for row in rows:
            if row.get("text"):
                continue                        # live-index rows arrive with their text
            row["text"] = (texts.get(row["doc_id"], "") or "")[:400]
        return source

    # ---------------- public API ----------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        stock_filter: Optional[List[str]] = None,
        cutoff_date: Optional[str] = None,
        use_decay: bool = True,
        interleave: bool = True,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Return (rows, error). Rows carry doc_id, date, stocks, text, label, score."""
        self._lazy_load()
        if not self._available:
            return [], self._error or "RAG tool unavailable"
        if not query or not query.strip():
            return [], "Empty query"

        k = top_k or self.top_k_fallback
        stock_list = [s.strip().upper() for s in (stock_filter or []) if s and s.strip()]
        stock_set = set(stock_list) or None
        limit = max(k * 3, k + 10)

        try:
            lexical = self._bm25_search(query, limit, cutoff_date, stock_set)
        except Exception as exc:
            return [], f"BM25 retrieval failed: {exc}"
        dense = self._dense_search(query, limit, cutoff_date, stock_list or None)
        live = self._live_search(query, limit, cutoff_date, stock_set)

        stock_filtered = bool(stock_set)
        if not lexical and not dense and not live and stock_set:
            # Macro themes are rarely tagged with the exact ETF we hold, so an
            # empty ticker-filtered result means "widen", not "no evidence".
            stock_filtered = False
            try:
                lexical = self._bm25_search(query, limit, cutoff_date, None)
            except Exception as exc:
                return [], f"BM25 retrieval failed: {exc}"
            dense = self._dense_search(query, limit, cutoff_date, None)
            live = self._live_search(query, limit, cutoff_date, None)

        if not lexical and not dense and not live:
            return [], ""  # nothing matched the filters; not an error

        meta: Dict[str, Dict[str, Any]] = {}
        recency: Dict[str, float] = {}
        relevance: Dict[str, float] = {}
        for ranked in (lexical, dense, live):
            for rank, row in enumerate(ranked):
                doc_id = row["doc_id"]
                meta.setdefault(doc_id, row)
                base = 1.0 / (rank + RRF_K)
                relevance[doc_id] = relevance.get(doc_id, 0.0) + base
                decayed = self._decay(base, row.get("date", ""), cutoff_date) if use_decay else base
                recency[doc_id] = recency.get(doc_id, 0.0) + decayed

        by_recency = sorted(recency, key=lambda d: -recency[d])
        by_relevance = sorted(relevance, key=lambda d: -relevance[d])

        selected: List[Tuple[str, str]] = []
        seen = set()
        if interleave and use_decay:
            # Barbell: alternate "what just happened" with "what happened last time".
            i = j = 0
            while len(selected) < k and (i < len(by_recency) or j < len(by_relevance)):
                if i < len(by_recency):
                    doc_id = by_recency[i]; i += 1
                    if doc_id not in seen:
                        seen.add(doc_id); selected.append((doc_id, "RECENT"))
                if len(selected) >= k:
                    break
                if j < len(by_relevance):
                    doc_id = by_relevance[j]; j += 1
                    if doc_id not in seen:
                        seen.add(doc_id); selected.append((doc_id, "HISTORICAL"))
        else:
            ordering = by_recency if use_decay else by_relevance
            for doc_id in ordering[:k]:
                selected.append((doc_id, "RECENT" if use_decay else "RELEVANT"))

        rows = []
        for doc_id, label in selected:
            row = dict(meta[doc_id])
            row["label"] = label
            row["score"] = round(recency.get(doc_id, 0.0) if use_decay else relevance.get(doc_id, 0.0), 6)
            row["ticker_filtered"] = stock_filtered
            rows.append(row)

        self._text_source = self._hydrate(rows)
        return rows, ""

    def status(self) -> Dict[str, Any]:
        self._lazy_load()
        if not self._available:
            mode = "unavailable"
        elif self.dense_enabled:
            mode = "hybrid_dense_bm25"
        elif self._store.available:
            mode = "bm25_local_text"
        else:
            mode = "bm25_metadata_only"
        return {
            "available": self._available,
            "mode": mode,
            "error": self._error,
            "notes": list(self._notes),
            "dense_enabled": self.dense_enabled,
            "live_index_rows": self.live_index.count() if self.live_index is not None else 0,
            "headline_store": str(self.headline_db_path) if self._store.available else None,
            "headline_rows": self._store.count() if self._store.available else 0,
            "bm25_index_path": str(self.bm25_index_path),
            "vector_store_path": str(self.vector_store_path),
        }
