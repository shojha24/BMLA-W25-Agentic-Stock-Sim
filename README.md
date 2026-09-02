# Agentic Stock Simulation

A multi-agent, news-driven forecasting panel. Three LLM personas read the same news digest,
retrieve historical analogues from a 876k-headline archive, argue with each other, and emit a
single consensus forecast.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# one-time: rebuild the local headline text store (~1 min, ~90 MB)
python rag_prep/build_headline_store.py

cp .env.example .env      # then paste your OpenRouter key into it

# offline smoke test - no API key, no network
python "Agent persona/src/main.py" --task roundtable --mode mock --quiet

# the real thing
python "Agent persona/src/main.py" --task roundtable --mode llm
```

`.env` must use the `VAR=value` form. A file containing only a bare key is silently
ignored by python-dotenv, which is why the LLM path used to fail with "API key not found".

## What runs today

| Piece | Status |
|---|---|
| 3 distinct LLM personas (macro / momentum / contrarian) | ✅ `Agent persona/src/agents/personas.py` |
| Agents exchange views and revise (2-round roundtable) | ✅ `orchestration/roundtable.py` |
| Confidence-weighted consensus + disagreement metrics | ✅ `orchestration/consensus.py` |
| Scorable forecast schema (direction, bps, horizon, confidence) | ✅ `core/types.py`, validated in `core/schema.py` |
| Hybrid RAG (BM25 + optional dense) with point-in-time cutoff, ticker filter, recency decay | ✅ `tools/rag.py` |
| Offline mode (no API key) for tests and CI | ✅ `--mode mock` |
| Live news ingestion every 15 minutes | ❌ digest is still a fixture (`Agent persona/data/digest.json`) |
| Market data, portfolio simulation, scheduler | ❌ not built |
| Benchmarks vs index / naive models, scoring | ❌ not built |

See `Agent persona/README.md` for the architecture and the CLI reference.

## Retrieval

The archive (2009-2020 analyst-ratings headlines) is indexed two ways:

* **BM25** - `dataset/news_bm25_index/`, committed to the repo.
* **Dense** - ChromaDB + Gemini embeddings in `dataset/vector_store/`, distributed
  out-of-band (Google Drive, ask Sharabh). Absent by default.

Because the vector store holds the headline *text*, a fresh clone used to retrieve doc_ids with
no content. `rag_prep/build_headline_store.py` fixes that: doc_id is `md5("date|stock|title")`,
so the text is rebuilt locally from the CSVs into SQLite. Check which tier is live with:

```bash
python "Agent persona/src/main.py" --task rag-status
```

Modes, best to worst: `hybrid_dense_bm25` → `bm25_local_text` → `bm25_metadata_only`.

## Research scripts (root)

Standalone experiments; the agent-facing port of this logic lives in `Agent persona/src/tools/rag.py`.

* `rag_core.py` - `EmbeddingManager`, `VectorStore`, `RAGRetriever`.
* `search.py` - BM25 vs dense vs RRF, with timings.
* `search_w_filter_decay.py` - adds date/ticker filters and recency decay.
* `search_interleaved.py` - adds the recent/historical "barbell" interleave.

`rag_prep/` holds the ingestion pipeline (see `rag_prep/README.md`); those scripts expect to be
run from the repo root and do not need re-running.

## Tests

```bash
python -m pytest "Agent persona/tests" -q
```

29 tests, no network required.
