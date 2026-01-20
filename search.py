import os
from google import genai
from google.genai import types
from rag_core import VectorStore

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "models/gemini-embedding-001"

class FreeTierEmbeddingManager:
    def __init__(self, api_key, model_name):
        """
        Initializes the new Google GenAI Client (v1.0)
        """
        if not api_key or "YOUR_API_KEY" in api_key:
            raise ValueError("Invalid API Key. Please set your GOOGLE_API_KEY.")
            
        # The new SDK uses a Client object
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def encode_query(self, text):
        """
        Embeds a single query string using the Free Tier limits.
        """
        try:
            # New SDK Syntax
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY" # Vital for accurate search results
                )
            )
            
            # Access the vector. The new SDK returns an object, not a dict.
            return response.embeddings[0].values
            
        except Exception as e:
            print(f"\nEmbedding Error: {e}")
            return []

def main():
    print("Initializing RAG System (Free Tier via google-genai)...")
    
    try:
        embedding_manager = FreeTierEmbeddingManager(API_KEY, MODEL_NAME)
    except Exception as e:
        print(e)
        return
    
    # 1. Connect to Database
    # We rely on rag_core only for the VectorStore class to access ChromaDB
    try:
        vector_store = VectorStore() 
        count = vector_store.collection.count()
        print(f"Connected to database. Total documents: {count}")
    except Exception as e:
        print(f"Database Error: {e}")
        return

    if count == 0:
        print("Warning: Vector store is empty! Run ingest.py first.")
        return

    # 2. Search Loop
    print("\nSystem Ready. Type 'exit' to quit.")
    while True:
        query = input("\nEnter query: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        try:
            # Step A: Embed the query (Free Tier)
            query_embedding = embedding_manager.encode_query(query)
            
            if not query_embedding:
                print("Failed to generate embedding. Check API quota?")
                continue
            
            # Step B: Search ChromaDB (Local)
            results = vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=10
            )
            
            print(f"\n--- Results for: {query} ---")
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    # safely get metadata
                    meta = results['metadatas'][0][i] or {}
                    content = results['documents'][0][i]
                    date = meta.get('date', 'N/A')
                    stock = meta.get('stock', 'N/A')
                    
                    print(f"[{date}] {stock}: {content}")
            else:
                print("No results found.")
                
        except Exception as e:
            print(f"Search failed: {e}")

if __name__ == "__main__":
    main()