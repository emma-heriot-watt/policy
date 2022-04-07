from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Optional


TEACH_ACTION_TO_SYNONYMS: Mapping[str, set[str]] = MappingProxyType(
    {
        "Forward": {"forward", "move ahead"},
        "Backward": {"backward"},
        "Turn Left": {"turn left"},
        "Turn Right": {"turn right"},
        "Look Up": {"look up"},
        "Look Down": {"look down"},
        "Pan Left": {"pan left", "strafe left"},
        "Pan Right": {"pan right", "strafe right"},
        "Move Up": {"move up"},
        "Move Down": {"move down"},
        "Pickup": {"pickup", "pick", "pick up", "lift"},
        "Place": {"place", "put", "put down", "drop"},
        "Open": {"open"},
        "Close": {"close"},
        "ToggleOn": {"toggle on", "switch on", "turn on"},
        "ToggleOff": {"toggle off", "switch off", "turn off"},
        "Slice": {"slice", "cut"},
        "Dirty": {"dirty"},
        "Clean": {"clean", "wash"},
        "Fill": {"fill"},
        "Empty": {"empty"},
        "Pour": {"pour"},
        "Break": {"break", "smash"},
    }
)


@lru_cache(maxsize=1)
def get_synonyms_to_teach_action_map() -> dict[str, str]:
    """Convert synonyms per action into a map to make it easier to get the correct action."""
    return {
        synonym: action
        for action, synonym_set in TEACH_ACTION_TO_SYNONYMS.items()
        for synonym in synonym_set
    }


@dataclass
class AgentAction:
    """A class that represents a robot action performed in the environment."""

    action: str
    object_label: Optional[str] = None
    object_visual_token: Optional[str] = None
