from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

from core.types import Digest, State, AgentOutput


class BaseAgent(ABC):
    name: str
    persona: str

    @abstractmethod
    def run(self, digest: Digest, state: State) -> AgentOutput:
        raise NotImplementedError