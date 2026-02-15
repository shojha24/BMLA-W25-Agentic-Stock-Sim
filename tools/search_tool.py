from typing import List, Optional
from search_interleaved import BacktestSearcher

# Initialize globally or within your agent context
rag_searcher = BacktestSearcher()

def tool_search_historical_news(
    query: str, 
    current_sim_date: str, 
    tickers: Optional[List[str]] = None
) -> str:
    """
    Tool for the agent to query historical news and context prior to making a trade.
    
    Args:
        query: The search term or question (e.g., "interest rate hikes impact on tech").
        current_sim_date: The current date in the simulation (format: YYYY-MM-DD). 
                          Prevents the agent from seeing future news.
        tickers: Optional list of specific stock tickers to filter by (e.g., ["AAPL", "QQQ"]).
        
    Returns:
        A formatted string of interleaved recent and historical news snippets.
    """
    # rag_searcher.search() now returns a fully formatted string
    formatted_results = rag_searcher.search(
        query=query,
        cutoff_date=current_sim_date,
        stock_list=tickers,
        use_decay=True,
        top_k=5  # Keep context window manageable
    )
    
    # If the search returned no documents, it will just be the headers. 
    # We can add a quick check to give the LLM a clearer "No Results" message if needed,
    # or just return the string exactly as is.
    if "1. [" not in formatted_results:
        return "No relevant historical context found for this query."
        
    return formatted_results