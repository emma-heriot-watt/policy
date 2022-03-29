from typing import Literal, Union

import torch
from emma_datasets.datamodels.datasets.teach import (
    ExtendedTeachDriverAction,
    TeachDriverAction,
    TeachEdhInstance,
)
from overrides import overrides

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem, EmmaVisualFeatures
from emma_policy.datamodules.pretrain_dataset import split_action_name
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.utils import get_logger


logger = get_logger(__name__)


class TeachEdhDataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for EDH instances from TEACh.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get the EDH instance at the given index as an instance of `EmmaDatasetItem`."""
        with self.db:
            instance_str: str = self.db[index]

        instance = TeachEdhInstance.parse_raw(instance_str)
        return self._convert_instance_to_emma_dataset_item(instance)

    def _convert_instance_to_emma_dataset_item(
        self, instance: TeachEdhInstance
    ) -> EmmaDatasetItem:
        """Convert the EDH instance to an instance of `EmmaDatasetItem`."""
        input_encoding = self.tokenizer(
            self._get_input_text_from_instance(instance),
            return_tensors=self._return_tensor_type,
            truncation=True,
        )
        target_encoding = self.tokenizer(
            self._get_target_text_from_instance(instance),
            return_tensors=self._return_tensor_type,
        )

        visual_features, scene_temporal_ids, object_temporal_ids = self._prepare_visual_input(
            instance
        )

        return EmmaDatasetItem(
            # Language
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            # Visual features
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            scene_temporal_ids=scene_temporal_ids,
            object_temporal_ids=object_temporal_ids,
            # Task
            task=self._get_task_as_tensor(Task.action_execution),
        )

    def _get_input_text_from_instance(self, instance: TeachEdhInstance) -> str:
        """Get the input text from a TEACh EDH instance."""
        dialog_history = self._get_concatenated_dialog_history(instance)
        actions = self._convert_actions_to_tokenizable_strings(
            instance.extended_driver_action_history,
            truncation_side="left",  # keep most recent actions
        )
        input_text = " ".join(dialog_history + actions)
        #  Add action execution task prefix
        input_text = self._get_random_template_for_task(Task.action_execution).format(
            instruction=input_text,
        )
        return input_text

    def _get_target_text_from_instance(self, instance: TeachEdhInstance) -> str:
        """Get the target text from a TEACh EDH instance."""
        actions_as_list = self._convert_actions_to_tokenizable_strings(
            instance.driver_actions_future,
            truncation_side="right",  # keep first actions
        )
        return " ".join(actions_as_list)

    def _get_concatenated_dialog_history(
        self, instance: TeachEdhInstance, cleaned: bool = True
    ) -> list[str]:
        """Get dialog history as a concatenated list of strings."""
        if cleaned:
            dialog_history = instance.dialog_history_cleaned
        else:
            dialog_history = instance.dialog_history

        concat_dialog_history = [
            f"<<{utterance.speaker.lower()}>> {utterance.utterance}"
            for utterance in dialog_history
            if utterance.utterance
        ]
        concat_dialog_history.append(self.tokenizer.sep_token)
        return concat_dialog_history

    def _convert_actions_to_tokenizable_strings(
        self,
        actions: Union[list[ExtendedTeachDriverAction], list[TeachDriverAction]],
        truncation_side: Literal["left", "right"] = "left",
    ) -> list[str]:
        """Convert actions from each TEACh EDH instance to tokenizable strings."""
        language: list[str] = []

        # Make sure to keep the same actions as frames
        if self.max_frames:
            actions = self._truncate_frames(actions, truncation_side=truncation_side)

        for action in actions:
            language.extend(split_action_name(action.action_name))
            language.append(self.tokenizer.sep_token)

        return language

    def _make_image_temporal_ids(
        self, feature_len_history: int, feature_len_future: int, object_frame_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get temporal ids for scenes and objects.

        We assign -1 to history tokens and the corresponding frame index to future tokens.
        """
        scene_temporal_ids = torch.cat(
            [torch.full((feature_len_history,), -1), torch.arange(1, feature_len_future + 1)]
        )
        # Relying on the fact that frame ids are consecutive tokens
        start = self.tokenizer("<frame_token_1>").input_ids[1]
        # We get the object frame id from the frame tokens
        object_frame_ids = object_frame_tokens - start + 1
        object_temporal_ids = object_frame_ids - feature_len_history
        object_temporal_ids.masked_fill_(object_temporal_ids <= 0, -1)
        return scene_temporal_ids, object_temporal_ids

    def _prepare_visual_input(
        self, instance: TeachEdhInstance
    ) -> tuple[EmmaVisualFeatures, torch.Tensor, torch.Tensor]:
        """Load history and future visual features and compute temporal ids."""
        # Load history visual features
        visual_features = self._load_visual_features(
            features_path=instance.features_path,
            modality=instance.modality,
            truncation_side="left",
            allow_empty=True,
        )
        len_history = visual_features.scene_features.shape[0]
        # Load future visual features
        len_future = 0
        if instance.future_features_path.exists():
            visual_features_future = self._load_visual_features(
                features_path=instance.future_features_path,
                modality=instance.modality,
                truncation_side="right",
                start_offset=len(visual_features.scene_features),
            )
            len_future = visual_features_future.scene_features.shape[0]
            # Combine history and future visual features
            visual_features = self._concat_visual_features(
                [visual_features, visual_features_future]
            )

        scene_temporal_ids, object_temporal_ids = self._make_image_temporal_ids(
            feature_len_history=len_history,
            feature_len_future=len_future,
            object_frame_tokens=visual_features.object_frame_tokens,
        )

        return visual_features, scene_temporal_ids, object_temporal_ids
