import dataclasses
from pathlib import Path
from typing import Union

import torch
from emma_datasets.datamodels.datasets.simbot import (
    SimBotClarificationTypes,
    SimBotInstructionInstance,
    SimBotQA,
)
from overrides import overrides
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem, EmmaVisualFeatures
from emma_policy.utils import get_logger


logger = get_logger(__name__)


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
        merged_annotations: bool = True,
        is_train: bool = True,
    ) -> None:

        if not merged_annotations:
            raise NotImplementedError(
                "Expecting dbs where every instance is an image associated with all of its question-answer pairs."
            )

        super().__init__(
            dataset_db_path=dataset_db_path, tokenizer=tokenizer, max_frames=max_frames
        )

        self.question_type_to_id = {
            question_type: index
            for index, question_type in enumerate(SimBotClarificationTypes)
            if question_type != SimBotClarificationTypes.other
        }
        self.is_train = is_train
        if is_train:
            index_db_map, dataset_size = self._unpack_annotations()
            self.index_db_map = index_db_map
            self.dataset_size = dataset_size
        else:
            self.dataset_size = len(self.db)

    @overrides(check_signature=False)
    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return self.dataset_size

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get the SimBot NLU instance at the given index as an instance of `EmmaDatasetItem`."""
        if self.is_train:
            db_map = self.index_db_map[index]
            index = db_map["db_index"]
        with self.db:
            instance_str = self.db[index]
        instance = SimBotInstructionInstance.parse_raw(instance_str)
        target_text, question_type_labels = self._get_target_text(instance=instance)

        if self.is_train:
            return self.nlu_instance(
                instance,
                target_text=target_text[db_map["question_index"]],
                question_type_labels=question_type_labels,
            )
        return self.nlu_instance(
            instance, target_text=target_text, question_type_labels=question_type_labels
        )

    def nlu_instance(
        self,
        instance: SimBotInstructionInstance,
        target_text: Union[str, list[str]],
        question_type_labels: torch.Tensor,
    ) -> EmmaDatasetItem:
        """Process the instance for the Simbot NLU task."""
        source_text = f"Predict the system act: {instance.instruction.instruction}"
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        visual_features = self._load_visual_features(
            features_path=instance.features_path,
            modality=instance.modality,
            truncation_side="right",  # Do not remove this
        )
        visual_features = self._keep_visual_features_for_first_action(
            visual_features=visual_features, num_frames=len(instance.actions[0].color_images)
        )
        if self.is_train:

            target_encoding = self.tokenizer.encode_plus(
                target_text, return_tensors=self._return_tensor_type, truncation=True
            )
            raw_target = None
        elif isinstance(target_text, list):
            # will use this to compute the score for act/clarify prediction
            target_encoding = self.tokenizer.encode_plus(
                target_text[0], return_tensors=self._return_tensor_type, truncation=True
            )
            raw_target = {"references": target_text, "question_type_labels": question_type_labels}
        else:
            raise AssertionError("Validation/Test target_text should be a list.")

        target_token_ids = target_encoding.input_ids.squeeze(0)
        decoder_attention_mask = target_encoding.attention_mask.squeeze(0)

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_token_ids,
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
            raw_target=raw_target,
        )

    def _keep_visual_features_for_first_action(
        self, visual_features: EmmaVisualFeatures, num_frames: int
    ) -> EmmaVisualFeatures:
        visual_features_dict = {
            field.name: getattr(visual_features, field.name)[:num_frames]
            for field in dataclasses.fields(EmmaVisualFeatures)
        }
        return EmmaVisualFeatures(**visual_features_dict)

    def _get_target_text(
        self, instance: SimBotInstructionInstance
    ) -> tuple[list[str], torch.Tensor]:
        target_text, question_type_labels = self._get_nlu_questions(instance=instance)
        if not target_text:
            target_text = ["<act>"]

        return target_text, question_type_labels

    def _get_nlu_questions(
        self, instance: SimBotInstructionInstance
    ) -> tuple[list[str], torch.Tensor]:
        questions = []
        question_type_labels = torch.zeros(len(self.question_type_to_id), dtype=torch.int)
        if instance.instruction.question_answers is not None:
            for question in instance.instruction.question_answers:
                if self._skip_question(question):
                    continue
                questions.append(self._prepare_question_nlu_target(question))
                index = self.question_type_to_id[question.question_type]
                question_type_labels[index] = 1
        return questions, question_type_labels

    def _prepare_question_nlu_target(self, question: SimBotQA) -> str:
        question_as_target = f"<clarify><{question.question_type.name}>"
        if question.question_target:
            question_as_target = f"{question_as_target} {question.question_target}"
        return question_as_target

    def _skip_question(self, question: SimBotQA) -> bool:
        skip = not question.question_necessary
        if question.question_type == SimBotClarificationTypes.other:
            skip = True
        return skip

    def _num_necessary_questions(self, instance: SimBotInstructionInstance) -> int:
        return len(self._get_nlu_questions(instance)[0])

    def _unpack_annotations(
        self,
    ) -> tuple[dict[int, dict[str, int]], int]:
        """Unpack the annotations from the db."""
        db_size = len(self.db)
        unpacked2packed: dict[int, dict[str, int]] = {}
        offset = 0
        dataset_size = 0
        with self.db:
            for index in range(db_size):
                instance_str: str = self.db[index]
                instance = SimBotInstructionInstance.parse_raw(instance_str)
                num_questions = self._num_necessary_questions(instance)
                individual_instances = [instance for _ in range(max(num_questions, 1))]

                for num_question, _ in enumerate(individual_instances):
                    unpacked2packed[offset + index + num_question] = {
                        "db_index": index,
                        "question_index": num_question,
                    }
                    dataset_size += 1

                offset += len(individual_instances) - 1
        return unpacked2packed, dataset_size
