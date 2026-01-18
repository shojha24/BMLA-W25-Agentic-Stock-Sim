import csv
import gc
import time  # <--- Added
from langchain_core.documents import Document
from rag_core import EmbeddingManager, VectorStore

def batch_generator(file_path, content_col, metadata_cols, batch_size=5000):
    """Yields batches of documents directly from CSV to save RAM."""
    current_batch = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        print(f"Streaming documents from {file_path}...")
        
        for row in reader:
            content = row.get(content_col, "")
            if not content.strip():
                continue
                
            metadata = {col: (row.get(col) or "") for col in metadata_cols}
            doc = Document(page_content=content, metadata=metadata)
            current_batch.append(doc)
            
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
        
        if current_batch:
            yield current_batch

def main():
    # 1. Setup
    file_path = "dataset/analyst_ratings_processed.csv"
    
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