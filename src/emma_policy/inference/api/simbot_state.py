import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class RequestTurn(BaseModel):
    """The environment state from the latest turn."""

    features: list[dict[str, Any]]
    output: Optional[str] = None


class RequestUtterance(BaseModel):
    """Information about a single utterance."""

    role: Literal["user", "agent"]
    utterance: str
    intent: Literal["instruction", "clarify_question", "clarify_answer"]


class GenerateRequest(BaseModel):
    """The dialog and environment history passed to the SimBot models."""

    dialogue_history: list[RequestUtterance]
    environment_history: list[RequestTurn]
