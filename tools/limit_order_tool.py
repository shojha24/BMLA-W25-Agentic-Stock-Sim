# Placeholder for the queue that the CDA engine processes every tick
PENDING_SIMULATION_ORDERS = []

def tool_submit_limit_order(
    agent_id: str, 
    ticker: str, 
    side: str, 
    quantity: int, 
    limit_price: float
) -> str:
    """
    Tool for the agent to submit a firm limit order to the market.
    
    Args:
        agent_id: The ID of the trading agent.
        ticker: The stock ticker (e.g., "QQQ").
        side: "BUY" or "SELL".
        quantity: Number of shares.
        limit_price: The exact price to execute at or better.
    """
    if side.upper() not in ["BUY", "SELL"]:
        return "Error: Side must be 'BUY' or 'SELL'."
    if quantity <= 0:
        return "Error: Quantity must be greater than 0."

    order = {
        "agent_id": agent_id,
        "ticker": ticker.upper(),
        "side": side.upper(),
        "quantity": quantity,
        "limit_price": limit_price,
        "status": "PENDING_MATCH"
    }
    
    PENDING_SIMULATION_ORDERS.append(order)
    
    return f"Order successfully routed to exchange: {side} {quantity} {ticker} @ ${limit_price} LIMIT."