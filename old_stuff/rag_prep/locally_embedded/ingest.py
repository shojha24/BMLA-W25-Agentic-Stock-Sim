import csv
import gc
import time  # <--- Added
from langchain_core.documents import Document
from rag_core import EmbeddingManager, VectorStore
import bm25s
import Stemmer
import hashlib

# Helper to generate consistent IDs
def generate_id(content, date, stock):
    # Combine the unique parts into one string
    # We add separators ("|") to prevent edge-case overlaps
    unique_string = f"{date}|{stock}|{content}"
    
    # Hash the combined string
    return hashlib.md5(unique_string.encode()).hexdigest()


def batch_generator(file_path, content_col, metadata_cols, batch_size=5000):
    """Yields batches of documents directly from CSV to save RAM."""
    current_batch = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        print(f"Streaming documents from {file_path}...")

        for row in reader:
            content = row.get(content_col, "")
            date = row.get("date", "")
            stock = row.get("stock", "")
            
            if not content.strip():
                continue
                
            # 1. Generate Composite ID
            doc_id = generate_id(content, date, stock)
            
            # 2. Assign to Metadata (LangChain will usually use this ID if you pass it correctly)
            metadata = {col: (row.get(col) or "") for col in metadata_cols}
            metadata['doc_id'] = doc_id 
            
            # Important: Many vector stores accept an 'id' parameter explicitly
            doc = Document(page_content=content, metadata=metadata)
            # Store the ID in the doc object so you can access it easily later if needed
            doc.id = doc_id 
            
            current_batch.append(doc)
            
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
        
        if current_batch:
            yield current_batch

def build_bm25_index(file_path, content_col):
    """Builds a BM25 index efficiently using bm25s native tokenization."""
    print(f"Building BM25 index from {file_path}...")
    
    # 1. Read only the text needed for indexing to save RAM
    # We avoid storing the full row dicts if possible, or just store what is needed.
    corpus_texts = []
    doc_ids = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            content = row.get(content_col, "")
            date = row.get("date", "")
            stock = row.get("stock", "")
            
            if not content.strip():
                continue

            # 1. Generate EXACT SAME Composite ID
            doc_id = generate_id(content, date, stock)
            doc_ids.append(doc_id)
            
            # 2. Add text to corpus for tokenization
            corpus_texts.append(content)

    # 2. Use bm25s optimized tokenizer (Multithreaded & handles punctuation)
    stemmer = Stemmer.Stemmer("english")
    print("Tokenizing corpus...")
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer)

    # 3. Create and Index
    print("Indexing...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    # 4. Save
    # We pass 'corpus_texts' so the retriever can return the actual text later
    # If you need metadata, you can load it separately or pass a dict list here, 
    # but a list of strings is much lighter on RAM.
    retriever.save("dataset/news_bm25_index", corpus=doc_ids)
    print("BM25 index built and saved.")


def main():
    # 1. Setup
    file_path = "dataset/analyst_ratings_consolidated.csv"
    
    print("Initializing models...")
    init_start = time.perf_counter() # Global timer start
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    print(f"Initialization took {time.perf_counter() - init_start:.2f}s")
    
    # 2. Start Multiprocessing Pool
    pool = embedding_manager.start_pool()
    
    total_docs_processed = 0
    batch_size = 5000 

    try:
        for batch_docs in batch_generator(file_path, "title", ["date", "stock"], batch_size):
            
            # --- START BATCH TIMER ---
            batch_start = time.perf_counter()
            
            # Extract text
            texts = [doc.page_content for doc in batch_docs]
            
            # Generate Embeddings
            embeddings = embedding_manager.encode_batch(texts, pool=pool)
            
            # Write to Disk
            vector_store.add_documents(batch_docs, embeddings)
            
            # --- END BATCH TIMER ---
            batch_end = time.perf_counter()
            duration = batch_end - batch_start
            
            # Calculate metrics
            count = len(batch_docs)
            total_docs_processed += count
            docs_per_sec = count / duration
            
            # Updated Print Statement
            print(f"Batch saved: {count} docs | Time: {duration:.2f}s | Speed: {docs_per_sec:.0f} docs/s | Total: {total_docs_processed}")
            
            # Memory Cleanup
            del texts
            del embeddings
            del batch_docs
            gc.collect()
        
        # 3. Build BM25 Index
        bm25_index = build_bm25_index(file_path, "title")
        print("BM25 index built and saved.")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        print("Closing worker pool...")
        embedding_manager.stop_pool(pool)
        
        total_time = time.perf_counter() - init_start
        print(f"Ingestion finished. Total time: {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()