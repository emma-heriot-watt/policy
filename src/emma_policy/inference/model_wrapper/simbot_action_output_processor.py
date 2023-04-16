from typing import Optional

from emma_common.datamodels import EmmaExtractedFeatures


class SimBotActionPredictionProcessor:
    """Process SimBot Action predictions."""

    def __call__(self, prediction: str, frame_features: list[EmmaExtractedFeatures]) -> str:
        """Process the prediction."""
        prediction = self._special_robotic_arm_button_case(prediction, frame_features)
        return self._special_carrot_case(prediction, frame_features)

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

    def _get_visual_token_from_prediction(self, prediction: str) -> Optional[int]:
        if "<vis_token" in prediction:
            return int(prediction.split("<vis_token_")[-1].split(">")[0])
        return None
