from typing import Optional

from emma_common.datamodels import EmmaExtractedFeatures


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
        force_token: bool = False,
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

            prediction_after_sink = self._special_sink_case(
                instruction, prediction_after_machine, entity_labels, force_token
            )

            return prediction_after_sink
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
            return f"toggle robot arm <frame_token_1> <vis_token_{token_id}> {self._stop_token}."
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
                return f"toggle button <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."

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
                return f"toggle everything's a carrot machine <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."
            elif is_place_instruction:
                return f"place everything's a carrot machine <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."
        return prediction

    def _special_sink_case(
        self,
        instruction: Optional[str],
        prediction: str,
        entity_labels: Optional[list[str]],
        force_token: bool,
    ) -> str:
        if not force_token or instruction is None or entity_labels is None:
            return prediction

        is_clean_instruction = any(
            [
                "clean" in instruction,
                "rinse" in instruction,
                "wash" in instruction,
                "cleanse" in instruction,
            ]
        )

        is_fill_instruction = any(
            [
                "fill" in instruction,
            ]
        )

        frame_token_id = prediction.split("frame_token_")[1].split(">")[0]
        if is_clean_instruction and "sink" in entity_labels:
            token_id = entity_labels.index("sink") + 1
            return f"clean sink <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."

        if is_fill_instruction and "sink" in entity_labels:
            token_id = entity_labels.index("sink") + 1
            return f"fill sink <frame_token_{frame_token_id}> <vis_token_{token_id}> {self._stop_token}."
        return prediction
