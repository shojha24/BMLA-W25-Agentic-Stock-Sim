import os
import time
import chromadb
import numpy as np
from typing import List, Dict, Any
import random
from google import genai
from google.genai import types
from dotenv import load_dotenv
import concurrent.futures

# --- CONFIGURATION ---
load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_PATH = "dataset/vector_store"

MODEL_NAME = "gemini-embedding-001"  # Change as needed


class EmbeddingManager:
    def __init__(self, project_id: str, location: str, model_name: str = MODEL_NAME):
        if not project_id or project_id == "YOUR_PROJECT_ID_HERE":
            raise ValueError("Please set a valid PROJECT_ID")
            
        print(f"Connecting to Vertex AI (Project: {project_id}, Loc: {location})...")
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.model_name = model_name
        self.api_batch_limit = 100 

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        chunks = []
        for i in range(0, len(texts), self.api_batch_limit):
            chunks.append(texts[i : i + self.api_batch_limit])

        all_embeddings = [None] * len(chunks)

        # MATCH: 5000 docs / 100 limit = 50 chunks.
        # Set workers to 50 to process the entire file batch in one wave.
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            future_to_index = {
                executor.submit(self._call_api_with_retry, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    batch_embeddings = future.result()
                    all_embeddings[index] = batch_embeddings
                except Exception as e:
                    print(f"Chunk {index} failed permanently: {e}")
                    # Fill with zero vectors to avoid crashing, but log heavily
                    # In a production system, you might want to retry differently
                    zero_vec = [0.0] * 768 
                    all_embeddings[index] = [zero_vec for _ in range(len(chunks[index]))]

        # 3. Flatten the list of lists
        flat_list = [item for sublist in all_embeddings for item in sublist]
        return np.array(flat_list)

    def _call_api_with_retry(self, chunk, retries=5):
        for attempt in range(retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=chunk,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        title="Financial News" 
                    )
                )
                return [e.values for e in response.embeddings]
            except Exception as e:
                # If we hit a rate limit (429), sleep longer
                if "429" in str(e) or "Resource exhausted" in str(e):
                    base_sleep = 2 * (attempt + 1)
                    # FIX: Use random jitter instead of 'index'
                    jitter = random.uniform(0.1, 1.0) 
                    time.sleep(base_sleep + jitter)
                elif attempt == retries - 1:
                    raise e
                else:
                    time.sleep(1)


class VectorStore:
    def __init__(self, collection_name: str = "headlines", persist_directory: str = DB_PATH):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "News headline embeddings"}
        )

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        ids = [doc.id for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        texts = [doc.page_content for doc in documents]
        embeddings_list = embeddings.tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=texts
        )

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Embed the query
        response = self.embedding_manager.client.models.embed_content(
            model=self.embedding_manager.model_name,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )
        
        # Extract single embedding
        query_embedding = response.embeddings[0].values
        
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                retrieved.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1 - results['distances'][0][i]
                })
        return retrieved