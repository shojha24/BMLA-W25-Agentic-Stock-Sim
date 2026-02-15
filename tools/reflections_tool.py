from rag_core import VectorStore, EmbeddingManager
from google import genai

# Initialize a separate VectorStore purely for agent memory
memory_store = VectorStore(collection_name="agent_reflections")
emb_manager = EmbeddingManager(project_id="YOUR_PROJECT_ID", location="YOUR_LOCATION")

def tool_query_internal_reflections(agent_id: str, ticker: str) -> str:
    """
    Tool for the agent to retrieve its own past reasoning and micro-reflections for a specific stock.
    
    Args:
        agent_id: The ID of the current agent (to ensure they don't read other agents' minds).
        ticker: The stock ticker in question (e.g., "QQQ").
    """
    query_text = f"Why did I trade {ticker}?"
    
    # 1. Embed the query
    client = genai.Client(vertexai=True, project="YOUR_PROJECT_ID", location="YOUR_LOCATION")
    response = client.models.embed_content(
        model=emb_manager.model_name,
        contents=query_text,
        config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_emb = response.embeddings[0].values
    
    # 2. Query ChromaDB, strictly filtering for this agent and this ticker
    results = memory_store.collection.query(
        query_embeddings=[query_emb],
        n_results=3,
        where={
            "$and": [
                {"agent_id": agent_id},
                {"ticker": ticker}
            ]
        }
    )
    
    if not results['documents'] or not results['documents'][0]:
        return f"You have no historical memory or reflections recorded for {ticker}."
        
    # 3. Format output
    memory_log = f"--- PAST REFLECTIONS FOR {ticker} ---\n"
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        date = meta.get('date', 'Unknown Date')
        sentiment = meta.get('bullish_score', 'N/A')
        memory_log += f"[{date} | Sentiment: {sentiment}] {doc}\n"
        
    return memory_log