from rag_core import EmbeddingManager, VectorStore, RAGRetriever

# --- CONFIGURATION ---
# Match these to what you used in ingest.py
PROJECT_ID = "gen-lang-client-0726681372"  # <--- PASTE YOUR PROJECT ID HERE
LOCATION = "us-central1"

def main():
    print("Initializing RAG System...")
    
    # 1. Load existing resources
    # FIX: Pass the required Project ID and Location to the manager
    try:
        embedding_manager = EmbeddingManager(
            project_id=PROJECT_ID,
            location=LOCATION
        )
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        print("Did you forget to set the PROJECT_ID in search.py?")
        return

    vector_store = VectorStore() 
    
    # 2. Check if data exists
    count = vector_store.collection.count()
    print(f"Connected to database. Total documents: {count}")
    
    if count == 0:
        print("Warning: Vector store is empty! Please run 'ingest.py' first.")
        return

    # 3. Initialize Retriever
    retriever = RAGRetriever(vector_store, embedding_manager)
    
    # 4. Interactive Loop
    print("\nSystem Ready. Type 'exit' to quit.")
    while True:
        query = input("\nEnter query: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        try:
            results = retriever.retrieve(query, top_k=10)
            
            print(f"\n--- Results for: {query} ---")
            if not results:
                print("No relevant results found.")
            
            for res in results:
                # Handle cases where metadata might be missing keys to prevent crashes
                date = res['metadata'].get('date', 'N/A')
                stock = res['metadata'].get('stock', 'N/A')
                print(f"[{date}] {stock}: {res['content']}")
                
        except Exception as e:
            print(f"Search failed: {e}")

if __name__ == "__main__":
    main()