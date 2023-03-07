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

        prediction_after_robot_arm = self._special_robotic_arm_button_case(
            prediction, entity_labels
        )
        prediction_after_button = self._special_button_case(
            instruction, prediction_after_robot_arm, entity_labels
        )

        return prediction_after_button

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
                token_id = entity_labels.index(color_button) + 1
                return f"toggle button <frame_token_1> <vis_token_{token_id}> <stop>."

        return prediction
