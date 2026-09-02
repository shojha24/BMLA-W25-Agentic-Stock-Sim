from __future__ import annotations
from typing import Dict, List, Protocol


class ChatClient(Protocol):
    """Minimal chat interface every agent depends on.

    Implemented by OpenRouterClient (live) and MockChatClient (offline/tests),
    so the whole pipeline can run without an API key.
    """

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        ...
