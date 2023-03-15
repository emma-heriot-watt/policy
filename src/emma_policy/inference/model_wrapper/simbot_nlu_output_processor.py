from typing import Optional

from emma_common.datamodels import EmmaExtractedFeatures

from emma_policy.datamodules.simbot_nlu_dataset import SimBotNLUIntents


class SimBotNLUPredictionProcessor:
    """Process SimBot NLU predictions."""

    def __init__(self, valid_action_types: list[str], default_prediction: str) -> None:
        self.valid_action_types = valid_action_types
        self._default_prediction = default_prediction

    def __call__(
        self, instruction: str, prediction: str, frame_features: list[EmmaExtractedFeatures]
    ) -> str:
        """Process the prediction."""
        object_name = self._get_target_object(prediction)

        if object_name is None:
            return prediction

        class_labels = self._get_detected_objects(frame_features=frame_features)
        if prediction.startswith(SimBotNLUIntents.act_no_match.value):
            prediction = self._special_robotic_arm_button_case(
                prediction=prediction,
                class_labels=class_labels,
            )

            prediction = self._special_machine_case(
                instruction=instruction,
                prediction=prediction,
                class_labels=class_labels,
            )
        elif prediction.startswith(SimBotNLUIntents.act_too_many_matches.value):
            prediction = self._rule_based_ambiguity_check(
                prediction=prediction,
                class_labels=class_labels,
                object_name=object_name,
            )
        if prediction.startswith(SimBotNLUIntents.act.value):
            prediction = self._special_monitor_toggle_case(
                instruction=instruction,
                prediction=prediction,
                class_labels=class_labels,
            )
        new_prediction = self._overwrite_the_nlu_prediction(prediction, object_name)
        return new_prediction

    def _prediction_type_is_valid(self, prediction: str) -> bool:
        # Make sure to return a valid format
        prediction_type = prediction.split(" ")[0]
        return prediction_type in self.valid_action_types

    def _overwrite_the_nlu_prediction(self, prediction: str, object_name: Optional[str]) -> str:
        """Check if the predicted NLU output needs to be overwritten."""
        # If the predicted prediction is not valid return the default prediction
        if not self._prediction_type_is_valid(prediction):
            return self._default_prediction
        # For search intents only return <search> object_name
        if prediction.startswith(SimBotNLUIntents.search.value):
            return f"{SimBotNLUIntents.search.value} {self._get_target_object(prediction)}"
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
        class_labels = frame_features[0].entity_labels
        if class_labels is not None:
            class_labels = [label.lower() for label in class_labels]
        return class_labels

    def _rule_based_ambiguity_check(
        self, prediction: str, class_labels: Optional[list[str]], object_name: str
    ) -> str:
        """Change too_many_matches prediction if there is one detected object."""
        # For now, overwrite the NLU only if there are no multiples in front of you
        # So if there's only one object that you are looking at, assume no ambiguity
        if class_labels is None:
            return prediction

        found_objects = [object_class == object_name for object_class in class_labels]
        if sum(found_objects) == 1:
            prediction = self._default_prediction

        return prediction

    def _special_robotic_arm_button_case(
        self, prediction: str, class_labels: Optional[list[str]]
    ) -> str:
        if class_labels is None:
            return prediction
        if "button" in prediction and "robot arm" in class_labels:
            return self._default_prediction
        return prediction

    def _special_machine_case(
        self, instruction: str, prediction: str, class_labels: Optional[list[str]]
    ) -> str:
        if class_labels is None:
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
            "carrot" in instruction or "machine" in instruction or "maker" in instruction
        )

        is_valid_instruction = (
            is_place_instruction or is_toggle_instruction
        ) and is_carrot_machine_instruction

        if "everything's a carrot machine" in class_labels and is_valid_instruction:
            return self._default_prediction

        return prediction

    def _special_monitor_toggle_case(  # noqa: WPS212, WPS231
        self, instruction: str, prediction: str, class_labels: Optional[list[str]]
    ) -> str:
        is_toggle_instruction = any(
            [
                "toggle" in instruction,
                "activate" in instruction,
                "turn" in instruction,
                "switch" in instruction,
            ]
        )
        if class_labels is None or not is_toggle_instruction:
            return prediction

        laser_monitor_in_bbox = "laser monitor" in class_labels
        if "laser" in instruction:
            if laser_monitor_in_bbox:
                return self._default_prediction
            return "<act><no_match> laser monitor"

        freeze_ray_monitor_in_bbox = "freeze ray monitor" in class_labels
        if "freeze" in instruction:
            if freeze_ray_monitor_in_bbox:
                return self._default_prediction
            return "<act><no_match> freeze ray monitor"

        gravity_flipper_monitor_in_bbox = "gravity monitor" in class_labels
        gravity_flipper_in_instruction = any(["gravity" in instruction, "flipper" in instruction])
        if gravity_flipper_in_instruction:
            if gravity_flipper_monitor_in_bbox:
                return self._default_prediction
            return "<act><no_match> gravity monitor"

        embiggenator_monitor_in_bbox = "embiggenator monitor" in class_labels
        if "embiggenator" in instruction:
            if embiggenator_monitor_in_bbox:
                return self._default_prediction
            return "<act><no_match> embiggenator monitor"

        is_portal_generator = "portal" in instruction or "generator" in instruction
        portal_generator_monitor_in_bbox = "portal generator monitor" in class_labels
        if is_portal_generator:
            if portal_generator_monitor_in_bbox:
                return self._default_prediction
            return "<act><no_match> portal generator monitor"

        return prediction
