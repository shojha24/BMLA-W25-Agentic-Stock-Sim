"""Rebuild the local headline text store from the source CSVs.

Run from the repo root:  python rag_prep/build_headline_store.py

Recovers doc_id -> headline text without needing the ChromaDB vector store,
by recomputing the same md5 doc_id that rag_prep/ingest.py assigned.
Takes ~1 minute for the 876k-row archive.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Agent persona" / "src"))

from tools.headline_store import HeadlineStore  # noqa: E402

CSV_PATHS = [
    ROOT / "dataset" / "analyst_ratings_consolidated_part1.csv",
    ROOT / "dataset" / "analyst_ratings_consolidated_part2.csv",
]
DB_PATH = ROOT / "dataset" / "headline_store.sqlite"


def main() -> None:
    print(f"Building headline store -> {DB_PATH}")
    n = HeadlineStore.build(CSV_PATHS, DB_PATH)
    if n == 0:
        print("No rows written. Are the dataset CSVs present?")
        raise SystemExit(1)
    store = HeadlineStore(DB_PATH)
    print(f"Done. {store.count():,} headlines available offline.")


if __name__ == "__main__":
    main()
