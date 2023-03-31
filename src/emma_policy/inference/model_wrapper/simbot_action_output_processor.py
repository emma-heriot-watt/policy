from typing import Optional

from emma_common.datamodels import EmmaExtractedFeatures


class SimBotActionPredictionProcessor:
    """Process SimBot Action predictions."""

    def __call__(self, prediction: str, frame_features: list[EmmaExtractedFeatures]) -> str:
        """Process the prediction."""
        return self._special_robotic_arm_button_case(prediction, frame_features)

    def _get_detected_objects(
        self, frame_features: list[EmmaExtractedFeatures]
    ) -> Optional[list[str]]:
        """Get a list of class labels fro the detected objects."""
        class_labels = frame_features[0].class_labels
        if class_labels is not None:
            class_labels = [label.lower() for label in class_labels]
        return class_labels

    def _special_robotic_arm_button_case(
        self, prediction: str, frame_features: list[EmmaExtractedFeatures]
    ) -> str:
        class_labels = self._get_detected_objects(frame_features)
        if class_labels is None:
            return prediction
        if "robot arm" in class_labels and "button" in prediction:
            token_id = class_labels.index("robot arm") + 1
            return f"toggle robot arm <frame_token_1> <vis_token_{token_id}> <stop>."
        return prediction