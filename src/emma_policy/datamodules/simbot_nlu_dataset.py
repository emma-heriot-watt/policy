import random
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from emma_datasets.constants.simbot.simbot import get_arena_definitions
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_from_action_object_metadata,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.paraphrasers import (
    InstructionParaphraser,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    SimBotAction,
    SimBotClarificationTypes,
    SimBotInstructionInstance,
    SimBotObjectAttributes,
    SimBotQA,
)
from overrides import overrides
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem, EmmaVisualFeatures
from emma_policy.datamodules.simbot_action_dataset import get_simbot_instruction_paraphrase
from emma_policy.utils import get_logger


logger = get_logger(__name__)


class SimBotNLUIntents(Enum):
    """SimBot NLU intent types."""

    act = "act"
    clarify = "clarify"


class SimBotNLUDataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for AreanNLU.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_frames: int = 4,
        is_train: bool = True,
        allow_paraphrasing: bool = False,
    ) -> None:
        super().__init__(
            dataset_db_path=dataset_db_path, tokenizer=tokenizer, max_frames=max_frames
        )
        self.question_type_to_id = {
            SimBotClarificationTypes.location: 0,
            SimBotClarificationTypes.disambiguation: 1,
            SimBotClarificationTypes.description: 2,
            SimBotClarificationTypes.direction: 3,
        }
        self.is_train = is_train
        self.data_intents: list[SimBotNLUIntents] = []
        if is_train:
            self._prepare_data_intents()

        arena_definitions = get_arena_definitions()
        self._object_assets_to_names = arena_definitions["asset_to_label"]
        self._allow_paraphrasing = allow_paraphrasing
        self._paraphraser = InstructionParaphraser()

    @overrides(check_signature=False)
    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return len(self.db)

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get the SimBot NLU instance at the given index as an instance of `EmmaDatasetItem`."""
        with self.db:
            instance_str = self.db[index]
        instance = SimBotInstructionInstance.parse_raw(instance_str)
        source_text = self._get_source_text(instance)
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_text, question_type_labels = self._get_target_text(instance=instance)
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        frame_idx = 0
        if instance.synthetic or "location" not in target_text:
            action_object_metadata = instance.actions[0].get_action_data.get("object", None)
            if action_object_metadata is not None:
                frame_idx = action_object_metadata.get("colorImageIndex", 0)
        visual_features = self._load_visual_features(
            features_path=instance.features_path[0], frame_idx=frame_idx
        )

        raw_target = {
            "example_id": f"{instance.mission_id}_{instance.annotation_id}_{instance.instruction_id}",
            "references": target_text,
            "question_type_labels": question_type_labels,
        }

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            raw_target=raw_target,
        )

    def _get_source_text(self, instance: SimBotInstructionInstance) -> str:
        """Process the source text for the Simbot NLU task."""
        if instance.actions[0].type == "Search":
            instruction = self._get_source_text_for_search(instance)

        else:
            instruction = self._get_source_text_for_low_level_action(instance)
        return f"Predict the system act: {instruction}"

    def _get_source_text_for_search(self, instance: SimBotInstructionInstance) -> str:
        """Get source text for Search instructions."""
        action_object_metadata = instance.actions[0].get_action_data["object"]
        object_candidates = len(action_object_metadata["id"])
        object_candidate_idx = random.choice(range(object_candidates))
        # Always paraphrase after selecting a target for search
        instruction = self._paraphraser(
            action_type="search",
            object_id=action_object_metadata["id"][object_candidate_idx],
            object_attributes=SimBotObjectAttributes(
                **action_object_metadata["attributes"][object_candidate_idx]
            ),
        )
        return instruction

    def _get_source_text_for_low_level_action(self, instance: SimBotInstructionInstance) -> str:
        """Get source text for all actions except Search."""
        if self._allow_paraphrasing and instance.paraphrasable:
            action_metadata = instance.actions[0].get_action_data["object"]
            object_name = get_object_from_action_object_metadata(
                object_asset=action_metadata["id"],
                object_assets_to_names=self._object_assets_to_names,
            )
            instruction = get_simbot_instruction_paraphrase(
                self._paraphraser, instance, object_name
            )
        else:
            instruction = instance.instruction.instruction
        return instruction

    def _get_target_text(self, instance: SimBotInstructionInstance) -> tuple[str, torch.Tensor]:
        # First try to get the target for a clarification
        target_text, question_type_labels = self._get_nlu_questions(instance=instance)
        # If target_text is an empty list, we have an action
        if not target_text:
            target_text = self._prepare_action_nlu_target(first_action=instance.actions[0])

        return target_text, question_type_labels

    def _prepare_action_nlu_target(self, first_action: SimBotAction) -> str:
        if first_action.type == "Search":
            target_text = "<act><search>"
        else:
            target_text = "<act><low_level>"
        return target_text

    def _get_nlu_questions(
        self, instance: SimBotInstructionInstance
    ) -> tuple[Optional[str], torch.Tensor]:
        """question_type_labels is a one-hot encoding of the question type."""
        question: Optional[str] = None
        question_type_labels = torch.zeros(len(self.question_type_to_id), dtype=torch.int)
        if not instance.synthetic and instance.instruction.question_answers is not None:
            question, question_type_labels = self._get_nlu_human_question(instance)
        elif instance.ambiguous:
            question, question_type_labels = self._get_nlu_synthetic_question(instance)

        return question, question_type_labels

    def _get_nlu_human_question(
        self, instance: SimBotInstructionInstance
    ) -> tuple[Optional[str], torch.Tensor]:
        """Get the target text and question type vector from a human question.

        Examples to avoid:
        1. water: Fill mug with water --> Where is the water?
        2. door: Close the fridge door/ --> Where is the door?
        3. Where is the office / room?
        """
        if len(instance.instruction.necessary_question_answers) > 1:
            raise AssertionError("Expected one question per instance.")
        question: Optional[str] = None
        question_type_labels = torch.zeros(len(self.question_type_to_id), dtype=torch.int)
        for qa_pair in instance.instruction.necessary_question_answers:
            if qa_pair.question_type not in self.question_type_to_id:
                continue
            if qa_pair.question_target in {"water", "door", "office", "room"}:
                continue
            question = self._prepare_question_nlu_target(qa_pair)
            index = self.question_type_to_id[qa_pair.question_type]
            question_type_labels[index] = 1

        return question, question_type_labels

    def _get_nlu_synthetic_question(
        self, instance: SimBotInstructionInstance
    ) -> tuple[str, torch.Tensor]:
        """Get the target text and question type vector from a synthetic question."""
        question_type = SimBotClarificationTypes.disambiguation
        question = f"<clarify><{question_type.name}>"
        action_type = instance.actions[-1].type
        action_metadata = getattr(instance.actions[-1], action_type.lower())
        object_metadata = action_metadata.get("object", None)["id"]
        if object_metadata is not None:
            question_target = get_object_from_action_object_metadata(
                object_metadata, self._object_assets_to_names
            )
            question = f"{question} {question_target.lower()}"
        # Question 1-hot encoding
        question_type_labels = torch.zeros(len(self.question_type_to_id), dtype=torch.int)
        question_type_labels[self.question_type_to_id[question_type]] = 1
        return question, question_type_labels

    def _prepare_question_nlu_target(self, question: SimBotQA) -> str:
        question_as_target = f"<clarify><{question.question_type.name}>"
        if question.question_target:
            question_as_target = f"{question_as_target} {question.question_target}"
        return question_as_target

    def _prepare_data_intents(self) -> None:
        """Prepare data intents for balancing."""
        db_size = len(self.db)
        with self.db:
            for index in range(db_size):
                instance_str: str = self.db[index]
                instance = SimBotInstructionInstance.parse_raw(instance_str)
                if instance.instruction.necessary_question_answers:
                    self.data_intents.append(SimBotNLUIntents.clarify)
                else:
                    self.data_intents.append(SimBotNLUIntents.act)

    @overrides(check_signature=False)
    def _load_visual_features(self, features_path: Path, frame_idx: int = 0) -> EmmaVisualFeatures:
        """Get the visual features just for the first frame."""
        frame_dict = torch.load(features_path)["frames"][frame_idx]
        feature_dicts = [frame_dict["features"]]

        visual_features = self._prepare_emma_visual_features(feature_dicts=feature_dicts)

        return visual_features
