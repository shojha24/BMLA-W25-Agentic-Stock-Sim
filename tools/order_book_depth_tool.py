from typing import Dict, Any

# Placeholder for the global Order Book managed by your CDA Simulator
GLOBAL_ORDER_BOOK = {
    "QQQ": {
        "bids": [(398.50, 100), (398.00, 250)], # (Price, Quantity)
        "asks": [(399.00, 50), (399.50, 200)]
    }
}

def tool_query_order_book_depth(ticker: str) -> str:
    """
    Tool for the agent to check the current market spread and liquidity for a ticker.
    
    Args:
        ticker: The stock ticker to look up (e.g., "QQQ").
    """
    book = GLOBAL_ORDER_BOOK.get(ticker)
    if not book:
        return f"No active order book data for {ticker}."
    
    # Format for the LLM
    highest_bid = book["bids"][0] if book["bids"] else ("None", 0)
    lowest_ask = book["asks"][0] if book["asks"] else ("None", 0)
    
    return (
        f"--- {ticker} ORDER BOOK ---\n"
        f"Highest Bid (Buyers): {highest_bid[1]} shares @ ${highest_bid[0]}\n"
        f"Lowest Ask (Sellers): {lowest_ask[1]} shares @ ${lowest_ask[0]}\n"
        f"Spread: ${(lowest_ask[0] - highest_bid[0]):.2f}"
    )