import json
import random
from pathlib import Path
from typing import Any

import torch
from emma_datasets.datamodels.datasets.simbot import SimBotAction, SimBotInstructionInstance
from overrides import overrides
from transformers import PreTrainedTokenizer

from emma_policy.common.settings import Settings
from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetItem,
    EmmaDatasetPadding,
    EmmaVisualFeatures,
)
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.utils import get_logger


settings = Settings()
ARENA_DICT_FILE = settings.paths.constants.joinpath("arena_definitions.json")


logger = get_logger(__name__)


def mask_past_target_actions(
    full_target_token_ids: torch.Tensor, sep_token_id: int, masking_value: int = -1
) -> torch.Tensor:
    """Mask the target token ids for all but the last action."""
    target_token_ids = full_target_token_ids.clone()
    separator_positions = torch.where(full_target_token_ids == sep_token_id)[0]
    if separator_positions.shape[0] > 1:
        end_index = int(separator_positions[-2].item()) + 1
        target_token_ids[:end_index] = masking_value  # noqa: WPS362
    return target_token_ids


class SimBotActionDataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for SimBotAction.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_frames: int = 15,
        use_only_necessary_questions: bool = True,
    ) -> None:

        super().__init__(
            dataset_db_path=dataset_db_path, tokenizer=tokenizer, max_frames=max_frames
        )

        self._use_only_necessary_questions = use_only_necessary_questions
        self._question_answer_prompts = [
            "With question: {question} and answer: {answer}.",
            "After asking: {question} and receiving: {answer}.",
            "With clarification: {question} and answer: {answer}.",
            "With question and answer: {question} {answer}.",
        ]
        with open(ARENA_DICT_FILE) as in_file:
            arena_constants = json.load(in_file)
            self._object_assets_to_names = arena_constants["asset_to_name"]

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get the `EmmaDatasetItem` for a SimBotInstructionInstance."""
        instance_str: str
        with self.db:
            instance_str = self.db[index]

        instance = SimBotInstructionInstance.parse_raw(instance_str)
        return self.simbot_action_execution(instance)

    def simbot_action_execution(self, instance: SimBotInstructionInstance) -> EmmaDatasetItem:
        """Process the instance for the SimBot action task."""
        if instance.instruction is None:
            raise AssertionError(
                "Instructions for this instance must exist. Make sure this instance is connected to the right task!"
            )
        actions_ids_in_instruction = instance.instruction.actions
        action_ids_in_instance = [action.id for action in instance.actions]
        if action_ids_in_instance != actions_ids_in_instruction:
            raise AssertionError(
                f"Instructions have {instance.instruction.actions} but found {action_ids_in_instance}."
            )

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )
        frames = []
        for fpath in instance.features_path:
            frames.extend([fdict["image"] for fdict in torch.load(fpath)["frames"]])

        source_text = self._prepare_source_text(instance=instance)

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        target_text = self._prepare_target_text(
            instance=instance, visual_features=visual_features, frames=frames
        )

        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        full_target_token_ids = target_encoding.input_ids.squeeze(0)

        target_token_ids = mask_past_target_actions(
            full_target_token_ids,
            sep_token_id=self.tokenizer.sep_token_id,  # type: ignore[arg-type]
            masking_value=EmmaDatasetPadding.target_token_ids,
        )
        decoder_input_ids = torch.full_like(
            full_target_token_ids,
            fill_value=self.tokenizer.eos_token_id,  # type: ignore[arg-type]
        )

        # Now shift them to the right
        decoder_input_ids[1:] = full_target_token_ids[:-1].clone()  # noqa: WPS362
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        raw_target = f"mission{instance.mission_id}_instr{instance.instruction_id}_ann{instance.annotation_id}_action{instance.actions[-1].type}"  # noqa: WPS221
        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_token_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.action_execution),
            raw_target=raw_target,
        )

    def _prepare_source_text(self, instance: SimBotInstructionInstance) -> str:
        """Prepare the source text.

        The source text allows the same template as the action execution task
        with the addition of the question and answer pair.

        Example:
            source_text = Execute the instruction: go to the desk with a hammer on it. With question
            and answer: where is the hammer? the hammer is on the table in the robotics lab.
        """
        source_text = self._get_random_template_for_task(Task.action_execution).format(
            instruction=instance.instruction.instruction
        )
        if instance.instruction.question_answers is not None:
            question_answer_candidates = instance.instruction.question_answers
            if self._use_only_necessary_questions:
                question_answer_candidates = [
                    qa for qa in instance.instruction.question_answers if qa.question_necessary
                ]
            if question_answer_candidates:
                question_answer = random.choice(question_answer_candidates)

                question_answer_prompt = random.choice(self._question_answer_prompts).format(
                    question=question_answer.question.lower(), answer=question_answer.answer
                )
                source_text = f"{source_text}. {question_answer_prompt}"

        if not source_text.endswith("."):
            source_text = f"{source_text}."
        source_text = source_text.replace("..", ".")

        return source_text

    def _prepare_target_text(  # noqa: WPS231
        self,
        instance: SimBotInstructionInstance,
        visual_features: EmmaVisualFeatures,
        frames: list[str],
    ) -> str:
        """Prepare the target text.

        The target text if the sequence of actions for the instruction instance separated with '.'.

        Example:
            target_text = Look Around. Goto Desk.
        """
        target_text = []
        for action in instance.actions:
            action_type = action.type
            action_metadata = getattr(action, action_type.lower())
            action_object_metadata = action_metadata.get("object", None)
            # case 1: navigation actions except GoTo
            if action_type in {"Look", "Move", "Rotate"}:
                target_text.append(f"{action_type} {action_metadata['direction']}.")
            # case 2: room/object navigation or interaction action
            elif action_object_metadata is not None:
                # action with a specific object
                if "id" in action_object_metadata:
                    object_name = self._get_object_from_action_object_metadata(
                        action_object_metadata
                    )
                    object_name_with_tokens = self._map_object_to_visual_token(
                        object_name=object_name,
                        action=action,
                        image_index=action_metadata["object"]["colorImageIndex"],
                        visual_features=visual_features,
                        frames=frames,
                    )
                    target_text.append(f"{action_type} {object_name_with_tokens}.")
                # action without an object (e.g, Goto Office)
                else:
                    # {'object': {'officeRoom': 'Lab1'}}
                    object_name = list(action_object_metadata.values())[0]
                    target_text.append(f"{action_type} {object_name}.")
            # no other action types, sanity check.
            else:
                raise AssertionError(f"Unsupported action {action}.")
        return " ".join(target_text).lower()

    def _get_object_from_action_object_metadata(
        self, action_object_metadata: dict[str, Any]
    ) -> str:
        """Map the object asset for a given action to its readable name.

        Example:
            (object_asset, object_name) = (Desk_01_1000, Desk)
        """
        object_asset = action_object_metadata["id"]
        # Case1: Object asset in action matches exactly with object assets
        object_name_candidate = self._object_assets_to_names.get(object_asset, None)
        if object_name_candidate is not None:
            return object_name_candidate

        # Case2: The object asset in action contains a substring that matches with the object assests
        # Example: Desk_01_1000
        # Because the assets can have additional tags we need to remove these tags
        # and check if they asset after removing the tags match an object label
        object_asset_components = object_asset.split("_")

        for idx in range(len(object_asset_components), 0, -1):
            # tries to match the longest sub-string first
            object_name_candidate = "_".join(object_asset_components[:idx])
            object_name_candidate = self._object_assets_to_names.get(object_name_candidate, None)
            if object_name_candidate is not None:
                return object_name_candidate
        return object_asset

    def _map_object_to_visual_token(
        self,
        object_name: str,
        action: SimBotAction,
        image_index: int,
        visual_features: EmmaVisualFeatures,
        frames: list[str],
    ) -> str:
        """Map the name of an object with its frame and visual token.

        If the object cannot be mapped return the object name alone.
        """
        # TODO: gpantaz update this to include feature and frame tokens
        color_image = action.color_images[image_index]
        frame_index = frames.index(color_image)
        scene_frame_token = self.tokenizer.decode(visual_features.scene_frame_tokens[frame_index])
        return f"{object_name} {scene_frame_token}"
