import csv
import gc
import time
import hashlib
import bm25s
import Stemmer
from langchain_core.documents import Document
from rag_core import EmbeddingManager, VectorStore

# --- GLOBAL SETTINGS ---
FILE_READ_BATCH_SIZE = 5000 
CSV_PATH = "dataset/analyst_ratings_consolidated.csv"

PROJECT_ID = "gen-lang-client-0726681372"  # <--- PASTE YOUR PROJECT ID HERE
LOCATION = "us-central1"

def generate_id(content, date, stock):
    unique_string = f"{date}|{stock}|{content}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def batch_generator(file_path, content_col, metadata_cols, batch_size=2000):
    current_batch = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        print(f"Streaming documents from {file_path}...")
        for row in reader:
            content = row.get(content_col, "")
            if not content.strip(): continue
            
            doc_id = generate_id(content, row.get("date",""), row.get("stock",""))
            metadata = {col: (row.get(col) or "") for col in metadata_cols}
            metadata['doc_id'] = doc_id
            
            doc = Document(page_content=content, metadata=metadata)
            doc.id = doc_id
            
            current_batch.append(doc)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
        if current_batch:
            yield current_batch

def build_bm25_index(file_path, content_col):
    print(f"\nBuilding BM25 index from {file_path}...")
    corpus_texts = []
    doc_ids = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            content = row.get(content_col, "")
            if not content.strip(): continue
            doc_id = generate_id(content, row.get("date",""), row.get("stock",""))
            doc_ids.append(doc_id)
            corpus_texts.append(content)

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save("dataset/news_bm25_index", corpus=doc_ids)
    print("BM25 index built and saved.")

def main():

    print("Initializing Google Vertex AI (Paid Tier)...")
    init_start = time.perf_counter()
    
    try:
        # Pass the Project ID here
        embedding_manager = EmbeddingManager(
            project_id=PROJECT_ID, 
            location=LOCATION
        )
    except Exception as e:
        print(f"Error init: {e}")
        print("Did you run 'gcloud auth application-default login' in your terminal?")
        return

    vector_store = VectorStore()
    print(f"Initialization took {time.perf_counter() - init_start:.2f}s")
    
    total_docs = 0

    try:
        for batch_docs in batch_generator(CSV_PATH, "title", ["date", "stock"], FILE_READ_BATCH_SIZE):
            
            batch_start = time.perf_counter()
            texts = [doc.page_content for doc in batch_docs]
            
            # This calls the updated logic in rag_core
            embeddings = embedding_manager.encode_batch(texts)
            
            vector_store.add_documents(batch_docs, embeddings)
            
            duration = time.perf_counter() - batch_start
            count = len(batch_docs)
            total_docs += count
            docs_per_sec = count / duration
            
            print(f"Batch saved: {count} docs | Time: {duration:.2f}s | Speed: {docs_per_sec:.0f} docs/s | Total: {total_docs}")
            
            del texts, embeddings, batch_docs
            gc.collect()

        build_bm25_index(CSV_PATH, "title")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        total_time = (time.perf_counter() - init_start) / 60
        print(f"Ingestion finished in {total_time:.2f} minutes.")

if __name__ == "__main__":
    main()