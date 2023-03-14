from collections import Counter
from typing import Optional

from emma_common.datamodels import EmmaExtractedFeatures


class SimBotActionPredictionProcessor:
    """Process SimBot Action predictions."""

    def __init__(self) -> None:
        self._button_colors = ["blue", "green", "red"]

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

            return prediction_after_special_monitor
        return prediction

    def _get_detected_objects(
        self, frame_features: list[EmmaExtractedFeatures]
    ) -> Optional[list[str]]:
        """Get a list of class labels fro the detected objects."""
        entity_labels = frame_features[0].entity_labels
        if entity_labels is not None:
            entity_labels = [label.lower() for label in entity_labels]
        return entity_labels

    def _special_robotic_arm_button_case(
        self, prediction: str, entity_labels: Optional[list[str]]
    ) -> str:
        if entity_labels is None:
            return prediction
        if "robot arm" in entity_labels and "button" in prediction:
            token_id = entity_labels.index("robot arm") + 1
            return f"toggle robot arm <frame_token_1> <vis_token_{token_id}> <stop>."
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
                return (
                    f"toggle button <frame_token_{frame_token_id}> <vis_token_{token_id}> <stop>."
                )

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
            ]
        )
        if not is_toggle_instruction:
            return prediction

        # Try to account for misclassification in the entity classifier.
        # If there is a single detected computer and the instruction is toggle, let the model act
        single_computer = Counter(entity_labels)["computer"] == 1
        # pickup bowl <frame_token_11> <vis_token_5> -> 11> 11> <vis_token_5> -> 11
        frame_token_id = prediction.split("frame_token_")[1].split(">")[0]

        laser_condition = "laser monitor" in entity_labels or single_computer
        if "laser" in instruction and laser_condition:
            try:
                token_id = entity_labels.index("laser monitor") + 1
            except ValueError:
                token_id = entity_labels.index("computer") + 1
            return f"toggle laser monitor <frame_token_{frame_token_id}> <vis_token_{token_id}> <stop>."

        freeze_ray_monitor_in_bbox = "freeze ray monitor" in entity_labels or single_computer
        if "freeze" in instruction and freeze_ray_monitor_in_bbox:
            try:
                token_id = entity_labels.index("freeze ray monitor") + 1
            except ValueError:
                token_id = entity_labels.index("computer") + 1
            return f"toggle freeze ray monitor <frame_token_{frame_token_id}> <vis_token_{token_id}> <stop>."

        gravity_flipper_monitor_in_bbox = "gravity monitor" in entity_labels or single_computer
        if "gravity" in instruction and gravity_flipper_monitor_in_bbox:
            try:
                token_id = entity_labels.index("gravity monitor") + 1
            except ValueError:
                token_id = entity_labels.index("computer") + 1
            return f"toggle gravity monitor <frame_token_{frame_token_id}> <vis_token_{token_id}> <stop>."

        embiggenator_monitor_in_bbox = "embiggenator monitor" in entity_labels or single_computer
        if "embiggenator" in instruction and embiggenator_monitor_in_bbox:
            try:
                token_id = entity_labels.index("embiggenator monitor") + 1
            except ValueError:
                token_id = entity_labels.index("computer") + 1
            return f"toggle embiggenator monitor <frame_token_{frame_token_id}> <vis_token_{token_id}> <stop>."

        is_portal_generator = "portal" in instruction or "generator" in instruction
        portal_generator_monitor_in_bbox = (
            "portal generator monitor" in entity_labels or single_computer
        )
        if is_portal_generator and portal_generator_monitor_in_bbox:
            try:
                token_id = entity_labels.index("portal generator monitor") + 1
            except ValueError:
                token_id = entity_labels.index("computer") + 1
            return f"toggle portal generator monitor <frame_token_{frame_token_id}> <vis_token_{token_id}> <stop>."
        return prediction
