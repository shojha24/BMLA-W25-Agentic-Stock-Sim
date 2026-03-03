from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


class RAGNewsTool:
    """
    Lightweight RAG retrieval utility for agent prompts.
    Uses BM25 for retrieval and optional Chroma lookup for full text.
    """

    def __init__(self, top_k_fallback: int = 8):
        self.top_k_fallback = top_k_fallback
        self._loaded = False
        self._available = False
        self._error = ""
        self._bm25 = None
        self._stemmer = None
        self._collection = None

        root = Path(__file__).resolve().parents[3]
        self.bm25_index_path = root / "dataset" / "news_bm25_index"
        self.vector_store_path = root / "dataset" / "vector_store"
        self.collection_name = "headlines"

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import bm25s  # type: ignore
            import Stemmer  # type: ignore
        except Exception as exc:
            self._error = f"RAG import failure (bm25s/Stemmer): {exc}"
            return

        if not self.bm25_index_path.exists():
            self._error = f"BM25 index not found at {self.bm25_index_path}"
            return

        try:
            self._bm25 = bm25s.BM25.load(str(self.bm25_index_path), load_corpus=True)
            self._stemmer = Stemmer.Stemmer("english")
        except Exception as exc:
            self._error = f"Failed to load BM25 index: {exc}"
            return

        try:
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=str(self.vector_store_path))
            self._collection = client.get_or_create_collection(name=self.collection_name)
        except Exception:
            # Chroma is optional; retrieval still works with metadata-only results.
            self._collection = None

        self._available = True

    @staticmethod
    def _parse_stocks(raw_stock: str) -> List[str]:
        if not raw_stock:
            return []
        return [s.strip() for s in str(raw_stock).split(",") if s.strip()]

    def _fetch_docs(self, doc_ids: List[str]) -> Dict[str, str]:
        if not doc_ids or not self._collection:
            return {}
        try:
            out = self._collection.get(ids=doc_ids)
            ids = out.get("ids", []) or []
            docs = out.get("documents", []) or []
            return {doc_id: (docs[i] or "") for i, doc_id in enumerate(ids)}
        except Exception:
            return {}

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        stock_filter: List[str] | None = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        self._lazy_load()
        if not self._available:
            return [], self._error or "RAG tool unavailable"
        if not query.strip():
            return [], "Empty query"

        try:
            import bm25s  # type: ignore
        except Exception as exc:
            return [], f"RAG import failure (bm25s): {exc}"

        k = top_k or self.top_k_fallback
        try:
            tokens = bm25s.tokenize(query, stemmer=self._stemmer)
            results, _ = self._bm25.retrieve(tokens, k=max(k * 3, k))
        except Exception as exc:
            return [], f"BM25 retrieval failed: {exc}"

        stock_set = {s.strip().upper() for s in (stock_filter or []) if s}
        rows: List[Dict[str, Any]] = []
        for item in results[0]:
            if isinstance(item, dict):
                doc_id = item.get("doc_id")
                date = item.get("date")
                stocks = self._parse_stocks(item.get("stock", ""))
            else:
                doc_id = str(item)
                date = ""
                stocks = []
            if not doc_id:
                continue

            if stock_set and stocks and not stock_set.intersection({s.upper() for s in stocks}):
                continue
            rows.append(
                {
                    "doc_id": doc_id,
                    "date": date,
                    "stocks": stocks,
                }
            )
            if len(rows) >= k:
                break

        doc_map = self._fetch_docs([row["doc_id"] for row in rows])
        for row in rows:
            text = doc_map.get(row["doc_id"], "")
            row["text"] = text[:400] if text else ""

        return rows, ""

    def status(self) -> Dict[str, Any]:
        self._lazy_load()
        return {
            "available": self._available,
            "error": self._error,
            "bm25_index_path": str(self.bm25_index_path),
            "vector_store_path": str(self.vector_store_path),
        }
