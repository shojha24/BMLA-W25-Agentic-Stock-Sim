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
    ) -> AgentOutput:
        """Produce one forecast set.

        `peer_context` / `prior_output` are supplied in round 2 of a roundtable
        so the agent can revise in light of what the other personas said.
        """
        raise NotImplementedError
