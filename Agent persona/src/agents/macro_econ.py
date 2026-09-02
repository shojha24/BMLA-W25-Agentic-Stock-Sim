"""Backwards-compatible shim.

The macro persona now lives in `agents/personas.py` and runs on the shared
`LLMPersonaAgent`. These names are kept so existing scripts and branches that
import from `agents.macro_econ` keep working.
"""
from __future__ import annotations

from typing import Optional

from agents.baseline import SentimentBaselineAgent
from agents.persona import LLMPersonaAgent
from agents.personas import MACRO_ECON
from llm.base import ChatClient
from tools.rag import RAGNewsTool

# `main.py --mode hardcoded` used to import this name; it never existed.
MacroEconomistAgent = SentimentBaselineAgent


class MacroEconomistLLMAgent(LLMPersonaAgent):
    def __init__(
        self,
        client: ChatClient,
        model: str = "minimax/minimax-m3:free",
        rag_tool: Optional[RAGNewsTool] = None,
        rag_top_k: int = 8,
        use_rag: bool = True,
    ):
        spec = MACRO_ECON
        if rag_top_k != spec.rag_top_k:
            spec = __import__("dataclasses").replace(spec, rag_top_k=rag_top_k)
        super().__init__(spec=spec, client=client, model=model, rag_tool=rag_tool, use_rag=use_rag)
