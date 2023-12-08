from typing import Optional, cast

import torch
from emma_common.datamodels import EmmaExtractedFeatures, EmmaPolicyRequest
from emma_common.logging import logger


def post_process_action(action: str) -> str:
    """Post process the action string.

    Remove the </s><s> at the begining of an instruction. Remove padding tokens. Keep other special
    tokens e.g, <vis_token_5>.
    """
    action = action.lstrip()
    action = action.replace("</s><s>", "")
    action = action.replace("<s>", "")
    action = action.replace("<pad>", "")
    return action


class SimBotActionPredictionProcessor:
    """Process SimBot Action predictions."""

    def __init__(self) -> None:
        self._button_colors = ["blue", "green", "red"]
        self._stop_token = "<stop>"  # noqa: S105

    def __call__(
        self,
        instruction: Optional[str],
        prediction: str,
        frame_features: list[EmmaExtractedFeatures],
    ) -> str:
        """Process the prediction."""
        entity_labels = self._get_detected_objects(frame_features)
        if instruction is None or entity_labels is None:
            return prediction

        return prediction

    def _get_detected_objects(
        self, frame_features: list[EmmaExtractedFeatures]
    ) -> Optional[list[str]]:
        """Get a list of class labels fro the detected objects."""
        class_labels = frame_features[-1].entity_labels
        if class_labels is not None:
            class_labels = [label.lower() for label in class_labels]
        return class_labels


class SimBotFindPredictionProcessor:
    """Process SimBot Find predictions."""

    def __call__(  # noqa: WPS231
        self, predicted_actions: list[str], simbot_request: EmmaPolicyRequest
    ) -> list[str]:
        """Process the prediction."""
        entity = simbot_request.entity_label
        post_processed_actions = []
        iterable = zip(predicted_actions, simbot_request.environment_history)
        for idx, (action, environment_state) in enumerate(iterable, 1):
            # Append only positive predictions, in case of no object return None
            if "token" in action:
                processed_action = post_process_action(action)
                # Fix the frame token in the case of multiple images
                processed_action = processed_action.replace(
                    "<frame_token_1>", f"<frame_token_{idx}>"
                )
                # Replace the <stop></s> at the end of the prediction
                # We know that the model has finished predicting in visual grounding.
                processed_action = processed_action.replace("<stop></s>", "").strip()
                post_processed_actions.append(processed_action)
            # Try to find the entity from the object labels
            elif entity is not None:
                entity_labels = [
                    entity_label.lower()
                    for entity_label in environment_state.features[0].entity_labels
                ]
                if entity_labels is not None and entity.lower() in entity_labels:
                    logger.debug(
                        "Policy didnt predict a visual token for the object but it was found in object labels"
                    )
                    largest_entity_idx = self._get_largest_entity(
                        entity, entity_labels, environment_state.features[0]
                    )

                    vis_token = largest_entity_idx + 1
                    post_processed_actions.append(f"<frame_token_{idx}> <vis_token_{vis_token}>")
        return post_processed_actions

    def _get_largest_entity(
        self, entity: str, entity_labels: list[str], features: EmmaExtractedFeatures
    ) -> int:
        indices = [
            idx for idx, entity_label in enumerate(entity_labels) if entity_label == entity.lower()
        ]
        bbox_coords = features.bbox_coords[indices]

        width = bbox_coords[:, 2] - bbox_coords[:, 0]
        height = bbox_coords[:, 3] - bbox_coords[:, 1]
        areas = width * height
        return indices[cast(int, torch.argmax(areas).item())]
