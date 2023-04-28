import re
from typing import Optional

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

        if "frame_token" in prediction and "vis_token" in prediction:
            prediction_after_robot_arm = self._special_robotics_lab_button_case(
                instruction, prediction, entity_labels
            )
            prediction_after_button = self._special_colorchanger_button_case(
                instruction, prediction_after_robot_arm, entity_labels
            )

            prediction_after_special_monitor = self._special_monitor_toggle_case(
                instruction, prediction_after_button, entity_labels
            )

            prediction_after_carrot = self._special_carrot_case(
                prediction_after_special_monitor, entity_labels
            )

            return prediction_after_carrot
        return prediction

    def _is_toggle_instruction(self, instruction: str) -> bool:
        return any(
            [
                " toggle " in instruction,
                " activate " in instruction,
                " turn " in instruction,
                " switch " in instruction,
                " flip " in instruction,
                " push " in instruction,
                " press " in instruction,
                " use " in instruction,
            ]
        )

    def _get_detected_objects(
        self, frame_features: list[EmmaExtractedFeatures]
    ) -> Optional[list[str]]:
        """Get a list of class labels fro the detected objects."""
        class_labels = frame_features[-1].entity_labels
        if class_labels is not None:
            class_labels = [label.lower() for label in class_labels]
        return class_labels

    def _special_robotics_lab_button_case(  # noqa: WPS231
        self, instruction: str, prediction: str, entity_labels: list[str]
    ) -> str:
        if "<stop>" not in prediction:
            return prediction

        is_toggle_instruction = self._is_toggle_instruction(instruction)
        button_in_instruction = "button" in instruction

        if is_toggle_instruction and button_in_instruction:
            frame_token_id = self._get_frame_token_from_prediction(prediction)
            token_id = None
            if "robot arm" in entity_labels:
                token_id = entity_labels.index("robot arm") + 1
                entity = "robot arm"
            elif "emotion tester" in entity_labels:
                token_id = entity_labels.index("emotion tester") + 1
                entity = "emotion tester"
            elif "printer" in entity_labels:
                token_id = entity_labels.index("printer") + 1
                entity = "printer"

            # TODO: check if we should only replace the prediction when no computer is present
            if token_id is not None and frame_token_id is not None:
                return f"toggle {entity} <frame_token_{frame_token_id}> <vis_token_{token_id}> <stop>."
        return prediction

    def _special_carrot_case(self, prediction: str, entity_labels: list[str]) -> str:
        """Remove the <stop> token whenever we are toggling the carrot machine.

        There is a bug in the arena where the agent gets a visual effects frame as the next frame
        whenever it tries to toggle the carrot machine. To handle this remove the stop token at the
        current time step and at the next timestep make a dummy action.
        """
        vis_token = self._get_visual_token_from_prediction(prediction)

        prediction_toggles_carrot_machine = (
            vis_token
            and entity_labels[vis_token - 1] == "everything's a carrot machine"
            and "toggle" in prediction
        )
        frame_token_id = self._get_frame_token_from_prediction(prediction)
        if prediction_toggles_carrot_machine and frame_token_id:
            return f"toggle everything's a carrot machine <frame_token_{frame_token_id}> <vis_token_{vis_token}>."

        # TODO: do we need force placing?
        tried_to_pick_up_carrot_machine = (
            vis_token
            and "pickup" in prediction
            and entity_labels[vis_token - 1] == "everything's a carrot machine"
        )
        if "carrot" in entity_labels and tried_to_pick_up_carrot_machine:
            new_vis_token = entity_labels.index("carrot") + 1
            return f"pick up carrot <frame_token_{frame_token_id}> <vis_token_{new_vis_token}> <stop>."
        return prediction

    def _special_colorchanger_button_case(
        self, instruction: str, prediction: str, entity_labels: list[str]
    ) -> str:
        if "<stop>" not in prediction:
            return prediction

        frame_token_id = self._get_frame_token_from_prediction(prediction)
        if frame_token_id is None:
            return prediction

        pattern = r".*(the )?(red|blue|green)( one| button)?\.$"
        match = re.search(pattern, instruction)
        if match is not None:
            color_result = re.search("(red|blue|green)", match.group())
            if color_result is not None:
                color = color_result.group()
                color_button = f"{color} button"
                if color is not None:
                    if color_button in entity_labels:
                        token_id = entity_labels.index(color_button) + 1  # noqa: WPS220
                        toggle_action = self._make_toggle(  # noqa: WPS220
                            "button", frame_token_id, token_id
                        )
                        return toggle_action  # noqa: WPS220

        return prediction

    def _special_monitor_toggle_case(  # noqa: WPS212, WPS231
        self, instruction: str, prediction: str, entity_labels: list[str]
    ) -> str:

        is_toggle_instruction = self._is_toggle_instruction(instruction)
        if not is_toggle_instruction or "<stop>" not in prediction:
            return prediction

        # pickup bowl <frame_token_11> <vis_token_5> -> 11> 11> <vis_token_5> -> 11
        frame_token_id = self._get_frame_token_from_prediction(prediction)
        if frame_token_id is None:
            return prediction

        laser_condition = "laser monitor" in entity_labels
        if "laser" in instruction and laser_condition:
            token_id = entity_labels.index("laser monitor") + 1
            return self._make_toggle("freeze ray monitor", frame_token_id, token_id)

        freeze_ray_monitor_in_bbox = "freeze ray monitor" in entity_labels
        if "freeze" in instruction and freeze_ray_monitor_in_bbox:
            token_id = entity_labels.index("freeze ray monitor") + 1
            return self._make_toggle("freeze ray monitor", frame_token_id, token_id)

        gravity_flipper_monitor_in_bbox = "gravity monitor" in entity_labels
        if "gravity" in instruction and gravity_flipper_monitor_in_bbox:
            token_id = entity_labels.index("gravity monitor") + 1
            return self._make_toggle("gravity monitor", frame_token_id, token_id)

        embiggenator_monitor_in_bbox = "embiggenator monitor" in entity_labels
        if "embiggenator" in instruction and embiggenator_monitor_in_bbox:
            token_id = entity_labels.index("embiggenator monitor") + 1
            return self._make_toggle("embiggenator monitor", frame_token_id, token_id)

        is_portal_generator = "portal" in instruction or "generator" in instruction
        portal_generator_monitor_in_bbox = "portal generator monitor" in entity_labels
        if is_portal_generator and portal_generator_monitor_in_bbox:
            token_id = entity_labels.index("portal generator monitor") + 1
            return self._make_toggle("portal generator monitor", frame_token_id, token_id)
        return prediction

    def _make_toggle(self, object_class: str, frame_token: int, vis_token: int) -> str:
        return f"toggle {object_class} <frame_token_{frame_token}> <vis_token_{vis_token}> {self._stop_token}."

    def _get_visual_token_from_prediction(self, prediction: str) -> Optional[int]:
        if "<vis_token" in prediction:
            return int(prediction.split("<vis_token_")[-1].split(">")[0])
        return None

    def _get_frame_token_from_prediction(self, prediction: str) -> Optional[int]:
        if "<frame_token" in prediction:
            return int(prediction.split("<frame_token_")[-1].split(">")[0])
        return None


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
        return indices[torch.argmax(areas).item()]  # type:ignore[call-overload]
