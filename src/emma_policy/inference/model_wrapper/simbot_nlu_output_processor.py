from typing import Optional

from emma_common.datamodels import EmmaExtractedFeatures

from emma_policy.datamodules.simbot_nlu_dataset import SimBotNLUIntents


class SimBotNLUPredictionProcessor:
    """Process SimBot NLU predictions."""

    def __init__(self, valid_action_types: list[str], default_prediction: str) -> None:
        self.valid_action_types = valid_action_types
        self._default_prediction = default_prediction

    def __call__(self, prediction: str, frame_features: list[EmmaExtractedFeatures]) -> str:
        """Process the prediction."""
        new_prediction = self._overwrite_the_nlu_prediction(prediction)
        if new_prediction != prediction:
            return new_prediction

        object_name = self._get_target_object(prediction)
        if object_name is None:
            return prediction
        if prediction.startswith(SimBotNLUIntents.act_no_match.value):
            return self._special_robotic_arm_button_case(
                prediction=prediction,
                frame_features=frame_features,
            )
        elif prediction.startswith(SimBotNLUIntents.act_too_many_matches.value):
            return self._rule_based_ambiguity_check(
                prediction=prediction,
                frame_features=frame_features,
                object_name=object_name,
            )
        return prediction

    def _prediction_type_is_valid(self, prediction: str) -> bool:
        # Make sure to return a valid format
        prediction_type = prediction.split(" ")[0]
        return prediction_type in self.valid_action_types

    def _overwrite_the_nlu_prediction(self, prediction: str) -> str:
        """Check if the predicted NLU output needs to be overwritten."""
        # If the predicted prediction is not valid return the default prediction
        if not self._prediction_type_is_valid(prediction):
            return self._default_prediction
        # For search intents only return <search>
        if prediction.startswith(SimBotNLUIntents.search.value):
            return SimBotNLUIntents.search.value
        # For act one_match intents only return <act><one_match>
        if prediction.startswith(SimBotNLUIntents.act_one_match.value):
            return SimBotNLUIntents.act_one_match.value
        return prediction

    def _get_target_object(self, prediction: str) -> Optional[str]:
        """Extract the target object from the NLU prediction."""
        split_parts = prediction.split(" ")
        return " ".join(split_parts[1:]) if len(split_parts) > 1 else None

    def _get_detected_objects(
        self, frame_features: list[EmmaExtractedFeatures]
    ) -> Optional[list[str]]:
        """Get a list of class labels fro the detected objects."""
        class_labels = frame_features[0].class_labels
        if class_labels is not None:
            class_labels = [label.lower() for label in class_labels]
        return class_labels

    def _rule_based_ambiguity_check(
        self, prediction: str, frame_features: list[EmmaExtractedFeatures], object_name: str
    ) -> str:
        """Change too_many_matches prediction if there is one detected object."""
        # For now, overwrite the NLU only if there are no multiples in front of you
        # So if there's only one object that you are looking at, assume no ambiguity
        class_labels = self._get_detected_objects(frame_features)
        if class_labels is None:
            return prediction

        found_objects = [object_class == object_name for object_class in class_labels]
        if sum(found_objects) == 1:
            prediction = self._default_prediction

        return prediction

    def _special_robotic_arm_button_case(
        self, prediction: str, frame_features: list[EmmaExtractedFeatures]
    ) -> str:
        class_labels = self._get_detected_objects(frame_features)
        if class_labels is None:
            return prediction
        if "button" in prediction and "robot arm" in class_labels:
            return self._default_prediction
        return prediction
