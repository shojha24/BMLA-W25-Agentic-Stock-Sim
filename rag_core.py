import os
import torch
import chromadb
import uuid
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Global settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DB_PATH = "dataset/vector_store"

class EmbeddingManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"): # ibm-granite/granite-embedding-english-r2
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device=DEVICE)
        
        # Optimization: Use FP16 on GPU
        if DEVICE == 'cuda':
            print("Model set to use fp16 for speed.")
            self.model.half()

    def start_pool(self):
        """Starts the multi-process pool."""
        if DEVICE == 'cuda':
            return self.model.start_multi_process_pool()
        return None

    def stop_pool(self, pool):
        """Stops the multi-process pool."""
        if pool:
            self.model.stop_multi_process_pool(pool)

    def encode_batch(self, texts: List[str], pool=None) -> np.ndarray:
        """Encodes a single batch, using the pool if provided."""
        if pool and DEVICE == 'cuda':
            return self.model.encode_multi_process(texts, pool, batch_size=512)
        else:
            return self.model.encode(texts, show_progress_bar=False, batch_size=512)

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
        # Generate IDs based on actual content to avoid duplicates if re-run (optional)
        # For now, we use UUIDs but ensuring we don't crash on duplicates
        ids = [f"doc_{uuid.uuid4()}" for _ in range(len(documents))]
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
        query_embedding = self.embedding_manager.encode_batch([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
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