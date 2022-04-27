import json
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Optional

import torch

from emma_policy.common.settings import Settings


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

AI2THOR_CLASS_DICT_FILE = Settings().paths.constants.joinpath("ai2thor_labels.json")


@lru_cache(maxsize=1)
def get_synonyms_to_teach_action_map() -> dict[str, str]:
    """Convert synonyms per action into a map to make it easier to get the correct action."""
    return {
        synonym: action
        for action, synonym_set in TEACH_ACTION_TO_SYNONYMS.items()
        for synonym in synonym_set
    }


@lru_cache(maxsize=1)
def load_teach_objects_to_indices_map() -> dict[str, int]:
    """Load teach object map dictionary."""
    with open(AI2THOR_CLASS_DICT_FILE) as in_file:
        object_indices_map = json.load(in_file)["label_to_idx"]
    return object_indices_map


@lru_cache(maxsize=1)
def get_lowercase_to_teach_object_map() -> dict[str, str]:
    """Map lowercase object names to teach objects."""
    with open(AI2THOR_CLASS_DICT_FILE) as in_file:
        object_indices_map = json.load(in_file)["label_to_idx"]
    lower_case_map = {object_name.lower(): object_name for object_name in object_indices_map}
    return lower_case_map


@dataclass
class AgentAction:
    """A class that represents a robot action performed in the environment."""

    action: str
    object_label: Optional[str] = None
    object_visual_token: Optional[str] = None
    object_to_index = load_teach_objects_to_indices_map()

    def get_object_index_from_visual_token(self) -> Optional[int]:
        """Get the index of the object - bounding box that matches the visual token.

        A visual token has the form <vis_token_X>. X is the index of the object starting with 1.
        """
        if self.object_visual_token is not None:
            return int(self.object_visual_token.split("_")[-1][0]) - 1
        return None

    def get_object_index_from_label(self, bbox_probas: torch.Tensor) -> Optional[int]:
        """Get the index of the object bounding box that matches the object label.

        If there are multiple objects with the same label, pick the one with the highest
        confidence.
        """
        if self.object_label is None:
            return None

        bbox_labels = torch.argmax(bbox_probas, -1)

        # TODO(george): check for lowercase in decoding
        object_index = self.object_to_index[self.object_label]
        object_index_in_bbox = torch.where(bbox_labels == object_index)[0]

        if len(object_index_in_bbox) > 1:
            object_probas = bbox_probas[object_index_in_bbox]
            most_confident_object_index = object_probas[:, object_index].argmax()
            return int(object_index_in_bbox[most_confident_object_index].item())

        return int(object_index_in_bbox[0].item())
