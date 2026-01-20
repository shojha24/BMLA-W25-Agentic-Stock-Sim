import time
import os
import bm25s
import Stemmer
from google import genai
from google.genai import types
from rag_core import VectorStore

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY") 
MODEL_NAME = "models/gemini-embedding-001"
BM25_INDEX_PATH = "dataset/news_bm25_index"

class RAGSearcher:
    def __init__(self):
        print("Loading resources...")
        if not API_KEY:
            raise ValueError("Missing GOOGLE_API_KEY environment variable.")
        self.client = genai.Client(api_key=API_KEY)
        
        self.vector_store = VectorStore()
        
        print("Loading BM25 Index...")
        # load_corpus=True is essential to get the objects back
        self.bm25 = bm25s.BM25.load(BM25_INDEX_PATH, load_corpus=True)
        self.stemmer = Stemmer.Stemmer("english")
        print("System Ready.")

    def embed_query(self, text):
        try:
            response = self.client.models.embed_content(
                model=MODEL_NAME,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def get_texts_by_ids(self, doc_ids):
        """
        Robust lookup: Checks Vector Store primary IDs, then Metadata.
        """
        if not doc_ids: return {}
        
        id_to_text = {}
        found_ids = set()

        # 1. Try Primary Key Lookup (Fastest)
        try:
            results = self.vector_store.collection.get(ids=doc_ids)
            for i, doc_id in enumerate(results['ids']):
                if results['documents'][i]:
                    id_to_text[doc_id] = results['documents'][i]
                    found_ids.add(doc_id)
        except Exception as e:
            print(f"Primary lookup warning: {e}")

        # 2. Metadata Lookup (Backup for mismatching IDs)
        missing_ids = [did for did in doc_ids if did not in found_ids]
        
        if missing_ids:
            try:
                # We search where metadata['doc_id'] is in our missing list
                meta_results = self.vector_store.collection.get(
                    where={"doc_id": {"$in": missing_ids}}
                )
                for i, doc in enumerate(meta_results['documents']):
                    # Get the ID stored in metadata to map it back correctly
                    stored_meta_id = meta_results['metadatas'][i].get('doc_id')
                    if stored_meta_id:
                        id_to_text[stored_meta_id] = doc
            except Exception as e:
                # Often happens if list is too big for a single query
                pass

        return id_to_text

    def search(self, query, top_k=10):
        print(f"\nProcessing: '{query}'...")
        
        # --- 1. RUN BM25 SEARCH ---
        bm25_start = time.perf_counter()
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        
        # Retrieve results
        bm25_raw_results, _ = self.bm25.retrieve(query_tokens, k=top_k)
        
        bm25_results_ids = []
        for item in bm25_raw_results[0]:
            # --- CRITICAL FIX: UNWRAP THE DICTIONARY ---
            if isinstance(item, dict):
                # We extract 'text' because that contains your MD5 hash
                doc_id = item.get('text') 
                if doc_id:
                    bm25_results_ids.append(doc_id)
            else:
                # Fallback if it is already a string
                bm25_results_ids.append(item)
                
        bm25_time = time.perf_counter() - bm25_start

        # --- 2. RUN VECTOR SEARCH ---
        vec_start = time.perf_counter()
        query_emb = self.embed_query(query)
        
        vector_resp = self.vector_store.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        vector_results_ids = vector_resp['ids'][0]
        vec_time = time.perf_counter() - vec_start

        # --- 3. COMPUTE RRF (Reciprocal Rank Fusion) ---
        rrf_scores = {}

        # We can now safely assume doc_id is a string (the hash)
        for rank, doc_id in enumerate(bm25_results_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (rank + 60))

        for rank, doc_id in enumerate(vector_results_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (rank + 60))

        rrf_results_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        # --- 4. DISPLAY ---
        all_ids = list(set(bm25_results_ids + vector_results_ids + rrf_results_ids))
        content_map = self.get_texts_by_ids(all_ids)

        print("\n" + "="*60)
        print(f"QUERY: {query}")
        print("="*60)

        self._print_section("MATCH 1: BM25 (Keyword)", bm25_results_ids, content_map, bm25_time)
        self._print_section("MATCH 2: GEMINI (Vector)", vector_results_ids, content_map, vec_time)
        self._print_section("MATCH 3: HYBRID (RRF Fusion)", rrf_results_ids, content_map, 0.0)

    def _print_section(self, title, doc_ids, content_map, duration):
        print(f"\n--- {title} [{duration:.4f}s] ---")
        for i, doc_id in enumerate(doc_ids):
            text = content_map.get(doc_id)
            if not text:
                text = "[Text lookup failed - ID mismatch]"
            
            # Clean up newlines for display
            text = text.replace("\n", " ")
            text_preview = (text[:85] + '..') if len(text) > 85 else text
            print(f"{i+1}. {text_preview}")

if __name__ == "__main__":
    try:
        searcher = RAGSearcher()
        while True:
            q = input("\nEnter query (or 'exit'): ")
            if q.strip().lower() in ['exit', 'quit']:
                break
            if not q.strip(): continue
            searcher.search(q)
    except KeyboardInterrupt:
        print("\nGoodbye.")