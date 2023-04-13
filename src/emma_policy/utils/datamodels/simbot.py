import random
from typing import Any, Union

import torch
from emma_datasets.datamodels.datasets.utils.simbot_utils.paraphrasers import (
    InstructionParaphraser,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    SimBotInstructionInstance,
    SimBotObjectAttributes,
)
from emma_datasets.db import DatasetDb

from emma_policy.utils import get_logger


logger = get_logger(__name__)


EMPTY_INVENTORY = "empty"


def mask_past_target_actions(
    full_target_token_ids: torch.Tensor, sep_token_id: int, masking_value: int = -1
) -> torch.Tensor:
    """Mask the target token ids for all but the last action."""
    target_token_ids = full_target_token_ids.clone()
    separator_positions = torch.where(full_target_token_ids == sep_token_id)[0]
    if separator_positions.shape[0] > 1:
        end_index = int(separator_positions[-2].item()) + 1
        target_token_ids[:end_index] = masking_value  # noqa: WPS362
    return target_token_ids


def get_simbot_instruction_paraphrase(
    paraphraser: InstructionParaphraser, instance: SimBotInstructionInstance, object_name: str
) -> str:
    """Paraphrase a SimBot instruction."""
    action_type = instance.actions[-1].type.lower()
    action_object_metadata = instance.actions[-1].get_action_data["object"]
    attributes = SimBotObjectAttributes(
        **action_object_metadata.get("attributes", {"readable_name": object_name})
    )
    return paraphraser(
        action_type=action_type,
        object_id=action_object_metadata["id"],
        object_attributes=attributes,
        inventory_object_id=instance.actions[-1].inventory_object_id,
    )


def format_instruction(text: str) -> str:
    """Make sure the instruction ends in a fullstop."""
    if not text.endswith(("?", ".")):
        text = f"{text}."
    text = text.replace("..", ".")
    return text.lower()


def compressed_mask_is_bbox(mask: Union[list[int], list[list[int]]]) -> bool:
    """Check if a compressed mask is a bounding box."""
    only_coords = len(mask) == 4
    not_list = not all([isinstance(x, list) for x in mask])
    return only_coords and not_list


def get_object_for_search(
    search_action: dict[str, Any],
    action_object_metadata: dict[str, Any],
    get_attributes: bool = True,
) -> tuple[str, int, dict[str, str]]:
    """Get the object id , index and attributes for a search instance."""
    # New version of visual augmentations
    selected_object_metadata = search_action.get("selected_object", None)
    attributes: dict[str, str] = {}
    if selected_object_metadata is not None:
        attributes = search_action["selected_object"]["attributes"]
        object_id = selected_object_metadata["id"]
        if object_id == "AP_Prop_Desk_Green" and object_id not in action_object_metadata["id"]:
            object_index = action_object_metadata["id"].index("AP_Prop_Desk_Green_model")
        else:
            object_index = action_object_metadata["id"].index(object_id)

    else:
        object_candidates = len(action_object_metadata["id"])
        object_index = random.choice(range(object_candidates))

        object_id = action_object_metadata["id"][object_index]
        if get_attributes:
            attributes = action_object_metadata["attributes"][object_index]

    return object_id, object_index, attributes


class SearchNegativeSampler:
    """Search negative selection class.

    Used to sample negative examples for the search objective. Creates a dictionary where keys
    indices of the positive examples in the dataset and values are the readable names of each
    object in the example. Given a readable name for an object, it samples keys from the dictionary
    until the readable name is not present in the list of objects for a key.
    """

    def __init__(self, db: DatasetDb):
        self._positive_indices_map = self._create_positive_indices_objects_map(db)

    def __call__(self, readable_name: str) -> int:
        """Sample a negative example."""
        while True:
            rand_idx = random.choice(list(self._positive_indices_map.keys()))
            if readable_name.lower() not in self._positive_indices_map[rand_idx]:
                return rand_idx

    def _create_positive_indices_objects_map(self, db: DatasetDb) -> dict[int, list[str]]:
        """Create a map of indices and positive examples."""
        db_size = len(db)
        positive_indices_map = {}
        with db:
            for index in range(db_size):
                instance_str: str = db[index]
                instance = SimBotInstructionInstance.parse_raw(instance_str)

                action = instance.actions[-1]
                if action.type == "Search" and action.search["object"]["mask"] is not None:
                    attributes = action.search["object"].get("attributes", None)
                    if attributes is not None:
                        readable_names = [  # noqa: WPS220
                            attribute["readable_name"].lower() for attribute in attributes
                        ]
                        positive_indices_map[index] = readable_names  # noqa: WPS220
        return positive_indices_map
