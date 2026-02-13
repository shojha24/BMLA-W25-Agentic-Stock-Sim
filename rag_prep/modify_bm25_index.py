# Ran only once because I realized my ingest logic for BM25 didn't allow for date/stock filtering by default, would've made things more annoying

import bm25s
from rag_core import VectorStore
import time

# --- CONFIG ---
BM25_PATH = "dataset/news_bm25_index"
BATCH_SIZE = 5000  # Number of IDs to look up at once

def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    print(f"Loading BM25 index from {BM25_PATH}...")
    # 1. Load the Index
    try:
        retriever = bm25s.BM25.load(BM25_PATH, load_corpus=True)
    except Exception as e:
        print(f"Failed to load index: {e}")
        return

    print(f"Index loaded. Total documents: {len(retriever.corpus)}")
    
    # 2. Connect to Vector Store (Source of Truth)
    print("Connecting to Vector Store...")
    vs = VectorStore()
    collection = vs.collection
    
    # 3. Prepare for Hydration
    new_corpus = []
    total_processed = 0
    start_time = time.time()
    
    # Extract just the ID strings from the current messy corpus
    # Handles both {'text': 'hash'} dicts and plain strings
    raw_ids = []
    for item in retriever.corpus:
        if isinstance(item, dict):
            # Curr format: {'id': 0, 'text': 'HASH'}
            raw_ids.append(item.get('text', ''))
        else:
            raw_ids.append(str(item))

    print(f"Starting metadata hydration for {len(raw_ids)} items...")

    # 4. Process in Batches
    for batch_ids in get_chunks(raw_ids, BATCH_SIZE):
        try:
            # Query ChromaDB for this batch of IDs
            # We only ask for 'metadatas' to save bandwidth
            results = collection.get(
                ids=batch_ids,
                include=['metadatas']
            )
            
            # Create a lookup map for this batch: { "hash_id": {metadata_dict} }
            meta_map = {}
            for i, doc_id in enumerate(results['ids']):
                meta = results['metadatas'][i] or {}
                meta_map[doc_id] = meta

            # Reconstruct the corpus objects for this batch
            # We preserve the ORIGINAL ORDER of 'batch_ids' strictly
            for doc_id in batch_ids:
                # Default values if Chroma doesn't have the ID (rare edge case)
                meta = meta_map.get(doc_id, {})
                
                new_item = {
                    "doc_id": doc_id,
                    "date": meta.get("date", "N/A"),
                    "stock": meta.get("stock", "N/A")
                }
                new_corpus.append(new_item)

            total_processed += len(batch_ids)
            if total_processed % 50000 == 0:
                print(f"Processed {total_processed} docs...")

        except Exception as e:
            print(f"Error processing batch: {e}")
            # If a batch fails, we must still append placeholders to keep alignment
            for doc_id in batch_ids:
                new_corpus.append({"doc_id": doc_id, "error": "lookup_failed"})

    # 5. Overwrite and Save
    print("Hydration complete. Saving new index...")
    retriever.corpus = new_corpus
    retriever.save(BM25_PATH)
    
    duration = time.time() - start_time
    print(f"Done! Updated {len(new_corpus)} documents in {duration:.2f}s.")
    print("Sample of new corpus format:", new_corpus[0])

if __name__ == "__main__":
    main()