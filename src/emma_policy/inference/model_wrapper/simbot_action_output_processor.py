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

        if "frame_token" in prediction and "vis_token" in prediction:
            prediction_after_robot_arm = self._special_robotic_arm_button_case(
                prediction, entity_labels
            )
            prediction_after_button = self._special_button_case(
                instruction, prediction_after_robot_arm, entity_labels
            )

            prediction_after_special_monitor = self._special_monitor_toggle_case(
                instruction, prediction_after_button, entity_labels
            )

            prediction_after_machine = self._special_machine_case(
                instruction, prediction_after_special_monitor, entity_labels
            )

            prediction_after_carrot = self._special_carrot_case(
                prediction_after_machine, frame_features
            )

            return prediction_after_carrot
        return prediction

    def _get_detected_objects(
        self, frame_features: list[EmmaExtractedFeatures]
    ) -> Optional[list[str]]:
        """Get a list of class labels fro the detected objects."""
        class_labels = frame_features[-1].entity_labels
        if class_labels is not None:
            class_labels = [label.lower() for label in class_labels]
        return class_labels

    def _special_robotic_arm_button_case(
        self, prediction: str, entity_labels: Optional[list[str]]
    ) -> str:
        if entity_labels is None:
            return prediction
        if "robot arm" in entity_labels and "button" in prediction:
            token_id = entity_labels.index("robot arm") + 1
            return f"toggle robot arm <frame_token_1> <vis_token_{token_id}> <stop>."
        return prediction

    def _special_carrot_case(
        self, prediction: str, frame_features: list[EmmaExtractedFeatures]
    ) -> str:
        """Remove the <stop> token whenever we are toggling the carrot machine.

        There is a bug in the arena where the agent gets a visual effects frame as the next frame
        whenever it tries to toggle the carrot machine. To handle this remove the stop token at the
        current time step and at the next timestep make a dummy action.
        """
        class_labels = self._get_detected_objects(frame_features)
        if class_labels is None:
            return prediction

        vis_token = self._get_visual_token_from_prediction(prediction)

        prediction_toggles_carrot_machine = (
            vis_token
            and class_labels[vis_token - 1] == "everything's a carrot machine"
            and "toggle" in prediction
        )
        if prediction_toggles_carrot_machine:
            return f"toggle everything's a carrot machine <frame_token_1> <vis_token_{vis_token}>."
        return prediction

    def _special_button_case(
        self, instruction: Optional[str], prediction: str, entity_labels: Optional[list[str]]
    ) -> str:
        if instruction is None or entity_labels is None:
            return prediction

        for color in self._button_colors:
            color_button = f"{color} button"
            should_modify_prediction = (
                color_button in entity_labels
                and color in instruction
                and "button" not in prediction
            )

            if should_modify_prediction:
                # pickup bowl <frame_token_11> <vis_token_5> -> 11> 11> <vis_token_5> -> 11
                frame_token_id = prediction.split("frame_token_")[1].split(">")[0]
                token_id = entity_labels.index(color_button) + 1
                # return f"toggle button <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."
                return self._make_toggle("button", frame_token_id, token_id)

        return prediction

    def _special_monitor_toggle_case(  # noqa: WPS212, WPS231
        self, instruction: Optional[str], prediction: str, entity_labels: Optional[list[str]]
    ) -> str:
        if instruction is None or entity_labels is None:
            return prediction

        is_toggle_instruction = any(
            [
                "toggle" in instruction,
                "activate" in instruction,
                "turn" in instruction,
                "switch" in instruction,
                "flip" in instruction,
            ]
        )
        if not is_toggle_instruction:
            return prediction

        # pickup bowl <frame_token_11> <vis_token_5> -> 11> 11> <vis_token_5> -> 11
        frame_token_id = prediction.split("frame_token_")[1].split(">")[0]

        laser_condition = "laser monitor" in entity_labels
        if "laser" in instruction and laser_condition:
            token_id = entity_labels.index("laser monitor") + 1
            return self._make_toggle("freeze ray monitor", frame_token_id, token_id)
            # return f"toggle laser monitor <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."

        freeze_ray_monitor_in_bbox = "freeze ray monitor" in entity_labels
        if "freeze" in instruction and freeze_ray_monitor_in_bbox:
            token_id = entity_labels.index("freeze ray monitor") + 1
            return self._make_toggle("freeze ray monitor", frame_token_id, token_id)
            # return f"toggle freeze ray monitor <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."

        gravity_flipper_monitor_in_bbox = "gravity monitor" in entity_labels
        if "gravity" in instruction and gravity_flipper_monitor_in_bbox:
            token_id = entity_labels.index("gravity monitor") + 1
            return self._make_toggle("gravity monitor", frame_token_id, token_id)
            # return f"toggle gravity monitor <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."

        embiggenator_monitor_in_bbox = "embiggenator monitor" in entity_labels
        if "embiggenator" in instruction and embiggenator_monitor_in_bbox:
            token_id = entity_labels.index("embiggenator monitor") + 1
            return self._make_toggle("embiggenator monitor", frame_token_id, token_id)
            # return f"toggle embiggenator monitor <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."

        is_portal_generator = "portal" in instruction or "generator" in instruction
        portal_generator_monitor_in_bbox = "portal generator monitor" in entity_labels
        if is_portal_generator and portal_generator_monitor_in_bbox:
            token_id = entity_labels.index("portal generator monitor") + 1
            return self._make_toggle("portal generator monitor", frame_token_id, token_id)
        return prediction

    def _make_toggle(self, object_class: str, frame_token: str, vis_token: int) -> str:
        return f"toggle {object_class} <frame_token_{frame_token}> <vis_token_{vis_token}> {self._stop_token}."

    def _special_machine_case(
        self, instruction: Optional[str], prediction: str, entity_labels: Optional[list[str]]
    ) -> str:
        if instruction is None or entity_labels is None:
            return prediction

        is_toggle_instruction = any(
            [
                "toggle" in instruction,
                "activate" in instruction,
                "turn" in instruction,
                "switch" in instruction,
            ]
        )

        is_place_instruction = any(
            [
                "place" in instruction,
                "put" in instruction,
            ]
        )

        is_carrot_machine_instruction = (
            "carrot machine" in instruction or "carrot maker" in instruction
        )

        # pickup bowl <frame_token_11> <vis_token_5> -> 11> 11> <vis_token_5> -> 11
        frame_token_id = prediction.split("frame_token_")[1].split(">")[0]
        if "everything's a carrot machine" in entity_labels and is_carrot_machine_instruction:
            token_id = entity_labels.index("everything's a carrot machine") + 1
            if is_toggle_instruction:
                return f"toggle everything's a carrot machine <frame_token_{frame_token_id}> <vis_token_{token_id}>."
            elif is_place_instruction:
                return f"place everything's a carrot machine <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."
        return prediction

    def _get_visual_token_from_prediction(self, prediction: str) -> Optional[int]:
        if "<vis_token" in prediction:
            return int(prediction.split("<vis_token_")[-1].split(">")[0])
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
