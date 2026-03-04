import csv
import hashlib
import bm25s
import Stemmer
from langchain_core.documents import Document
from rag_core import EmbeddingManager, VectorStore

# --- CONFIG ---
MAIN_CSV_PATH = "dataset/analyst_ratings_consolidated.csv"
MACRO_CSV_PATH = "dataset/macro_events.csv"  # <--- Single file source
BM25_PATH = "dataset/news_bm25_index"

# GCP Config
PROJECT_ID = "gen-lang-client-0726681372"
LOCATION = "us-central1"

def generate_id(content, date, stock):
    """MUST match the logic used in your original ingestion exactly."""
    unique_string = f"{date}|{stock}|{content}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def load_macro_events():
    """Reads the single macro CSV file."""
    print(f"Loading macro events from {MACRO_CSV_PATH}...")
    try:
        with open(MACRO_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            events = list(reader)
            print(f" -> Loaded {len(events)} macro events.")
            return events
    except FileNotFoundError:
        print(f" [!] Error: {MACRO_CSV_PATH} not found.")
        return []

def ingest_vectors(macro_events):
    """Embeds ONLY the new macro events and adds to Vector Store."""
    if not macro_events:
        print("No macro events to ingest.")
        return

    print(f"\n--- 1. Vector Injection ({len(macro_events)} items) ---")
    print("Initializing Embedding Manager...")
    
    # Init Embedding Manager
    try:
        embedding_manager = EmbeddingManager(project_id=PROJECT_ID, location=LOCATION)
    except Exception as e:
        print(f"Error init EmbeddingManager: {e}")
        return

    vector_store = VectorStore()
    
    docs = []
    texts = []
    
    for event in macro_events:
        # Generate ID
        doc_id = generate_id(event['title'], event['date'], event['stock'])
        
        # Prepare Metadata
        metadata = {
            "date": event['date'],
            "stock": event['stock'],
            "doc_id": doc_id,
            "source": "macro_history" # Useful tag for debugging later
        }
        
        # Create Document
        doc = Document(page_content=event['title'], metadata=metadata)
        doc.id = doc_id
        
        docs.append(doc)
        texts.append(event['title'])
    
    # Embed ONLY these items (Fast & Cheap)
    print(f"Generating embeddings for {len(texts)} macro headlines...")
    embeddings = embedding_manager.encode_batch(texts)
    
    # Add to existing Vector Store
    print("Upserting to Vector Store...")
    vector_store.add_documents(docs, embeddings)
    print("Success: Macro events added to Vector Store.")

def rebuild_bm25(macro_events):
    """
    Rebuilds BM25 index by reading Main CSV (Read-Only) + New Macro Data.
    Does NOT modify the Main CSV file.
    """
    print(f"\n--- 2. BM25 Rebuild (Consolidated + Macro) ---")
    
    corpus_texts = []
    corpus_records = []
    
    # A. Stream Main CSV (Read-Only)
    print(f"Reading main dataset from {MAIN_CSV_PATH}...")
    try:
        with open(MAIN_CSV_PATH, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                content = row.get("title", "")
                if not content.strip(): continue
                
                # Re-generate ID on the fly to match Vector Store
                doc_id = generate_id(content, row.get("date",""), row.get("stock",""))
                
                corpus_records.append({
                    "doc_id": doc_id,
                    "date": row.get("date", ""),
                    "stock": row.get("stock", "")
                })
                corpus_texts.append(content)
                
    except FileNotFoundError:
        print(f"Error: Could not find {MAIN_CSV_PATH}")
        return

    print(f" -> Loaded {len(corpus_texts)} existing documents.")

    # B. Add Macro Events
    print(f"Adding {len(macro_events)} macro events to index...")
    for event in macro_events:
        content = event['title']
        doc_id = generate_id(content, event['date'], event['stock'])
        
        corpus_records.append({
            "doc_id": doc_id,
            "date": event['date'],
            "stock": event['stock']
        })
        corpus_texts.append(content)

    # C. Tokenize & Index
    total_docs = len(corpus_texts)
    print(f"Tokenizing {total_docs} total documents...")
    
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer)
    
    print("Fitting BM25 algorithm...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    print(f"Saving new index to {BM25_PATH}...")
    retriever.save(BM25_PATH, corpus=corpus_records)
    print("Success: BM25 Index updated.")

def main():
    # 1. Load the new data
    macro_data = load_macro_events()
    
    if not macro_data:
        return

    # 2. Embed & Ingest (Only the new stuff)
    ingest_vectors(macro_data)
    
    # 3. Re-index Search (Everything, but fast)
    rebuild_bm25(macro_data)
    
    print("\nDONE. System is now macro-aware.")

if __name__ == "__main__":
    main()