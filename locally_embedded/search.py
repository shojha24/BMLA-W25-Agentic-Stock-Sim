from rag_core import EmbeddingManager, VectorStore, RAGRetriever

def main():
    print("Initializing RAG System...")
    
    # 1. Load existing resources
    # Since VectorStore uses PersistentClient, it simply connects to the existing folder
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore() 
    
    # 2. Check if data exists
    if vector_store.collection.count() == 0:
        print("Warning: Vector store is empty! Please run 'ingest.py' first.")
        return

    # 3. Initialize Retriever
    retriever = RAGRetriever(vector_store, embedding_manager)
    
    # 4. Interactive Loop
    print("\nSystem Ready. Type 'exit' to quit.")
    while True:
        query = input("\nEnter query: ")
        if query.lower() == 'exit':
            break
            
        results = retriever.retrieve(query, top_k=10)
        
        print(f"\n--- Results for: {query} ---")
        for res in results:
            print(f"[{res['metadata']['date']}] {res['metadata']['stock']}: {res['content']}")

if __name__ == "__main__":
    main()