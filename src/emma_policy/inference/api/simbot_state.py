import logging
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class SpeakerRole(Enum):
    """Speaker roles."""

    user = "user"
    agent = "agent"


SPEAKER_TOKEN_MAP: Mapping[SpeakerRole, str] = MappingProxyType(
    {SpeakerRole.user: "<<commander>>", SpeakerRole.agent: "<<driver>>"}
)


class RequestTurn(BaseModel):
    """The environment state from the latest turn."""

    features: list[dict[str, Any]]
    output: Optional[str] = None


class RequestUtterance(BaseModel):
    """Information about a single utterance."""

    role: SpeakerRole
    utterance: str


class GenerateRequest(BaseModel):
    """The dialog and environment history passed to the SimBot models."""

    dialogue_history: list[RequestUtterance]
    environment_history: list[RequestTurn]
