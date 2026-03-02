from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from livekit.agents import ProviderTool


class StepfunTool(ProviderTool, ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class WebSearch(StepfunTool):
    """Enable Stepfun's built-in web search tool."""

    def __post_init__(self) -> None:
        super().__init__(id="stepfun_web_search")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "web_search"}


@dataclass
class Retrieval(StepfunTool):
    """Enable Stepfun's retrieval tool with a vector store id."""

    vector_store_id: str

    def __post_init__(self) -> None:
        super().__init__(id="stepfun_retrieval")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "retrieval",
            "vector_store_id": self.vector_store_id,
        }
