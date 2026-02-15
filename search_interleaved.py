import time
import os
import copy
import math
from datetime import datetime
import bm25s
import Stemmer
from google import genai
from google.genai import types
from rag_core import VectorStore

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY") 
MODEL_NAME = "models/gemini-embedding-001"
BM25_INDEX_PATH = "dataset/news_bm25_index"

# Decay Factor: How much to penalize news per day?
# 0.995 means a story loses 0.5% relevance every day it gets older.
# After 1 year, a story retains only ~16% of its original score.
DECAY_RATE = 0.995

class BacktestSearcher:
    def __init__(self):
        print("Loading resources for Backtesting...")
        if not API_KEY: raise ValueError("Missing GOOGLE_API_KEY")
        self.client = genai.Client(api_key=API_KEY)
        
        # 1. Load Chroma
        self.vector_store = VectorStore()
        
        # 2. Load BM25 (Must be hydrated with metadata!)
        print("Loading BM25 Index...")
        self.bm25 = bm25s.BM25.load(BM25_INDEX_PATH, load_corpus=True)
        self.stemmer = Stemmer.Stemmer("english")
        print("System Ready.")

    def embed_query(self, text):
        try:
            response = self.client.models.embed_content(
                model=MODEL_NAME, contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def get_texts_and_meta_by_ids(self, doc_ids):
        """
        Fetch content AND metadata for display/decay calculation.
        """
        if not doc_ids: return {}
        id_map = {}
        
        try:
            results = self.vector_store.collection.get(ids=doc_ids)
            for i, doc_id in enumerate(results['ids']):
                if results['documents'][i]:
                    id_map[doc_id] = {
                        "text": results['documents'][i],
                        "metadata": results['metadatas'][i] or {}
                    }
        except: pass
        
        return id_map

    def calculate_recency_score(self, base_score, doc_date_str, cutoff_date_str):
        """
        Applies a time-decay function to the score.
        New Score = Old Score * (DECAY_RATE ^ Days_Old)
        """
        if not doc_date_str or doc_date_str == "N/A":
            return base_score 

        try:
            # --- FIX: Handle Long Timestamps (2015-12-24 09:40...) ---
            # We only care about the YYYY-MM-DD part (first 10 chars)
            clean_date_str = str(doc_date_str)[:10]
            
            # Parse Dates
            doc_dt = datetime.strptime(clean_date_str, "%Y-%m-%d")
            
            # If cutoff is None, use "Today" (simulation time)
            if cutoff_date_str:
                sim_dt = datetime.strptime(cutoff_date_str, "%Y-%m-%d")
            else:
                sim_dt = datetime.now()

            # Calculate Delta in Days
            delta = (sim_dt - doc_dt).days
            
            # Clamp negative delta (if doc is somehow in the future relative to cutoff)
            if delta < 0: delta = 0
            
            # Apply Exponential Decay
            decay_factor = math.pow(DECAY_RATE, delta)
            
            # Debug Print (Optional - helps verify it's working)
            # if delta < 365: print(f"  > Decay debug: {clean_date_str} (-{delta}d) factor={decay_factor:.4f}")
            
            return base_score * decay_factor

        except Exception as e:
            # If parsing still fails, print why so we aren't blind
            print(f"Date Parse Error: {e} on '{doc_date_str}'")
            return base_score

    def get_filtered_bm25_indices(self, cutoff_date=None, stock_list=None):
        """
        Returns list of integer indices for BM25 slicing.
        """
        indices = []
        for idx, item in enumerate(self.bm25.corpus):
            # item = {'doc_id':..., 'date':..., 'stock':...}
            
            # 1. Stock Filter (List check)
            if stock_list:
                doc_stock = item.get('stock')
                if doc_stock not in stock_list:
                    continue
            
            # 2. Date Filter (Hard Cutoff)
            if cutoff_date:
                doc_date = item.get('date')
                # If doc has no date, decide if you keep or toss. 
                # Safe option for backtesting: Toss it.
                if not doc_date or doc_date > cutoff_date:
                    continue
                    
            indices.append(idx)
        return indices

    def search(self, query, cutoff_date=None, stock_list=None, use_decay=True, top_k=10):
        print(f"\nProcessing: '{query}'")
        if cutoff_date: print(f"  [Backtest Mode] Cutoff: {cutoff_date}")
        if stock_list:  print(f"  [Stock Filter]  Targets: {stock_list}")
        if use_decay:   print(f"  [Decay Enabled] Rate: {DECAY_RATE}")

        # OVERSAMPLE: We fetch more results than needed to ensure we have enough
        # remaining after we throw away the "future" news.
        SAFE_K = 1000 

        # --- 1. BM25 SEARCH (Retrieve -> Filter) ---
        bm25_start = time.perf_counter()
        
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        # Fetch raw results
        results, _ = self.bm25.retrieve(query_tokens, k=SAFE_K)
        
        bm25_results_ids = []
        for item in results[0]:
            # item = {'doc_id': '...', 'date': '...', 'stock': '...'}
            
            # A. Check Stock Filter (Logic: Must be in list)
            if stock_list and item.get('stock') not in stock_list:
                continue
                
            # B. Check Date Filter (Logic: Must be <= cutoff)
            if cutoff_date:
                doc_date = item.get('date')
                if not doc_date or doc_date > cutoff_date:
                    continue
            
            bm25_results_ids.append(item['doc_id'])
            
            # Optimization: Stop once we have enough to fill the Top K
            if len(bm25_results_ids) >= top_k * 2:
                break
                
        bm25_time = time.perf_counter() - bm25_start

        # --- 2. VECTOR SEARCH (Retrieve Stock -> Filter Date) ---
        vec_start = time.perf_counter()
        query_emb = self.embed_query(query)
        
        # Build Chroma Where Clause (STOCKS ONLY)
        # We REMOVED the date check here to prevent the ValueError
        where_clause = {}
        if stock_list:
             # Using $in is standard, but some older Chroma versions prefer $or
             # Try this first. If it fails, let me know.
            where_clause = {"stock": {"$in": stock_list}}
        
        # Query Chroma
        # We ask for SAFE_K because we might throw many away based on date
        vector_resp = self.vector_store.collection.query(
            query_embeddings=[query_emb],
            n_results=SAFE_K,
            where=where_clause # Only filtering stocks here
        )
        
        # Filter Dates in Python
        vector_results_ids = []
        
        # We need to look at the metadata returned by Chroma to check the date
        raw_ids = vector_resp['ids'][0]
        raw_metas = vector_resp['metadatas'][0]
        
        for i, doc_id in enumerate(raw_ids):
            meta = raw_metas[i] or {}
            
            # Apply Date Filter
            if cutoff_date:
                doc_date = meta.get('date')
                # String comparison works fine in Python ("2020" > "2019")
                if not doc_date or doc_date > cutoff_date:
                    continue
            
            vector_results_ids.append(doc_id)
            if len(vector_results_ids) >= top_k * 2:
                break
                
        vec_time = time.perf_counter() - vec_start

        # --- 3. FETCH METADATA FOR DECAY ---
        all_ids = list(set(bm25_results_ids + vector_results_ids))
        content_map = self.get_texts_and_meta_by_ids(all_ids)

        # --- 4. CALCULATE DUAL SCORES ---
        # We maintain two separate scoreboards
        scores_decayed = {}
        scores_historical = {}

        def calculate_scores(doc_id, rank):
            # Base Relevance Score (RRF)
            base_score = 1 / (rank + 60)
            
            # 1. Historical Score (Pure Relevance, No Time Penalty)
            scores_historical[doc_id] = scores_historical.get(doc_id, 0) + base_score
            
            # 2. Decayed Score (Recency Bias)
            if use_decay:
                doc_data = content_map.get(doc_id)
                if doc_data:
                    date_str = doc_data['metadata'].get('date')
                    # Use your FIXED date parser here
                    decayed_score = self.calculate_recency_score(base_score, date_str, cutoff_date)
                else:
                    decayed_score = base_score
            else:
                decayed_score = base_score
                
            scores_decayed[doc_id] = scores_decayed.get(doc_id, 0) + decayed_score

        # Populate scores
        for rank, doc_id in enumerate(bm25_results_ids):
            calculate_scores(doc_id, rank)
        for rank, doc_id in enumerate(vector_results_ids):
            calculate_scores(doc_id, rank)

        # --- 5. INTERLEAVE RESULTS (The Barbell) ---
        # Get Top N "Breaking News"
        top_recent = sorted(scores_decayed, key=scores_decayed.get, reverse=True)
        
        # Get Top N "Historical Context"
        top_history = sorted(scores_historical, key=scores_historical.get, reverse=True)
        
        final_results = []
        seen_ids = set()
        
        # We want 'top_k' total, so we take top_k/2 from each list
        half_k = int(top_k / 2)
        
        # Helper to safely add unique docs
        def add_unique(doc_list, limit):
            count = 0
            for doc_id in doc_list:
                if count >= limit: break
                if doc_id not in seen_ids:
                    final_results.append(doc_id)
                    seen_ids.add(doc_id)
                    count += 1
        
        # Strategy: Alternate adding 1 recent, then 1 historical
        # This keeps the final list balanced
        ptr_rec = 0
        ptr_hist = 0
        
        while len(final_results) < top_k:
            # Add Recent
            if ptr_rec < len(top_recent):
                doc_id = top_recent[ptr_rec]
                if doc_id not in seen_ids:
                    final_results.append(doc_id)
                    seen_ids.add(doc_id)
                ptr_rec += 1
            
            if len(final_results) >= top_k: break
                
            # Add Historical
            if ptr_hist < len(top_history):
                doc_id = top_history[ptr_hist]
                if doc_id not in seen_ids:
                    final_results.append(doc_id)
                    seen_ids.add(doc_id)
                ptr_hist += 1
                
            # Break if we run out of docs in both lists
            if ptr_rec >= len(top_recent) and ptr_hist >= len(top_history):
                break

        # --- 6. RETURN ---
        ''' # DEBUG PRINTING (Optional) - Shows the final interleaved results with labels and dates
        print("\n" + "="*60)
        print(f"QUERY: {query}")
        print("="*60)
        
        print(f"\n--- HYBRID RESULTS (Barbell Strategy) ---")
        for i, doc_id in enumerate(final_results):
            data = content_map.get(doc_id)
            if data:
                text = data['text'][:85].replace("\n", " ") + "..."
                meta = data['metadata']
                date_display = meta.get('date', 'N/A')
                
                # Label the output so you know WHY it was picked
                # Simple heuristic: if it's in the top_recent list (high rank), it's "News"
                label = "HISTORICAL"
                if doc_id in top_recent[:half_k+2]: # loose check
                    label = "RECENT    "
                
                print(f"{i+1}. [{label}] [{date_display}] {text}")
        '''

        output_lines = []
        output_lines.append("=" * 60)
        output_lines.append(f"QUERY: {query}")
        output_lines.append("=" * 60)
        output_lines.append("\n--- HYBRID RESULTS (Barbell Strategy) ---")
        
        for i, doc_id in enumerate(final_results):
            data = content_map.get(doc_id)
            if data:
                # Note: Passing this to an LLM, so using full text, vs for printing where we truncated to 85 chars
                text = data['text'].replace("\n", " ")
                meta = data['metadata']
                date_display = meta.get('date', 'N/A')
                
                # Label the output so you know WHY it was picked
                label = "HISTORICAL"
                if doc_id in top_recent[:half_k+2]: # loose check
                    label = "RECENT    "
                
                output_lines.append(f"{i+1}. [{label}] [{date_display}] {text}")

        # Return the final aggregated string
        return "\n".join(output_lines)

if __name__ == "__main__":
    searcher = BacktestSearcher()
    
    # Example Backtest Scenario
    # "It is Jan 1st, 2019. I am trading NVDA. What is the news?"
    print(searcher.search(
        query="graphics card demand", 
        cutoff_date="2019-01-01", 
        stock_list=["NVDA", "AMD"], 
        use_decay=True
    ))
    # "It is June 15th, 2013. The market is panicking about the Fed. How are banks looking?"
    print(searcher.search(
        query="federal reserve tapering bond buying yields",
        cutoff_date="2013-06-15",
        stock_list=["JPM", "BAC", "C", "GS", "FED", "TRADE", "MACRO"],
        use_decay=True
    ))
    # "It is August 24th, 2015 (Black Monday). China markets are crashing. Who is exposed?"
    print(searcher.search(
        query="China economic slowdown yuan devaluation sales",
        cutoff_date="2015-08-24",
        stock_list=["AAPL", "YUM", "CAT", "GM"],
        use_decay=True
    ))
    # "It is April 15th, 2019. Disney just unveiled their streaming service. Is Netflix in trouble?"
    print(searcher.search(
        query="streaming service launch subscriber growth competition",
        cutoff_date="2019-04-15",
        stock_list=["NFLX", "DIS", "AAPL", "CMCSA"],
        use_decay=True
    ))