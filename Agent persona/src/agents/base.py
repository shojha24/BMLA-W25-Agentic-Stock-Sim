from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.types import AgentOutput, Digest, State


class BaseAgent(ABC):
    name: str
    persona: str

    @abstractmethod
    def run(
        self,
        digest: Digest,
        state: State,
        peer_context: Optional[List[Dict[str, Any]]] = None,
        prior_output: Optional[AgentOutput] = None,
        brief: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """Produce one forecast set and the orders that express it.

        `peer_context` / `prior_output` are supplied in round 2 of a roundtable
        so the agent can revise in light of what the other personas said.
        `brief` is the assembled 15-Minute Brief; when present it carries the
        news summary, the agent's own balance and trades, and the retrieved
        historical context, and the agent does not retrieve for itself.
        """
        raise NotImplementedError
