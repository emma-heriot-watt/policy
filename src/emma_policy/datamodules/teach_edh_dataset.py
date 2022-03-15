from typing import Union

from emma_datasets.datamodels.datasets.teach import (
    ExtendedTeachDriverAction,
    TeachDriverAction,
    TeachEdhInstance,
)

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.utils import get_logger


logger = get_logger(__name__)


class TeachEdhDataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for EDH instances from TEACh.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

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

        visual_features = self._load_visual_features(instance)

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
            # Task
            task=self._get_task_as_tensor(Task.action_execution),
        )

    def _get_input_text_from_instance(self, instance: TeachEdhInstance) -> str:
        """Get the input text from a TEACh EDH instance."""
        actions = self._convert_actions_to_tokenizable_strings(
            instance.extended_driver_action_history
        )
        return " ".join(actions)

    def _get_target_text_from_instance(self, instance: TeachEdhInstance) -> str:
        """Get the target text from a TEACh EDH instance, with task template."""
        actions_as_list = self._convert_actions_to_tokenizable_strings(
            instance.driver_actions_future
        )
        actions_as_string = " ".join(actions_as_list)
        return self._get_random_template_for_task(Task.action_execution).format(
            instruction=actions_as_string
        )

    def _convert_actions_to_tokenizable_strings(
        self, actions: Union[list[ExtendedTeachDriverAction], list[TeachDriverAction]]
    ) -> list[str]:
        """Convert actions from each TEACh EDH instance to tokenizable strings.

        The speaker for the actions is equivalent to what was used in the ET baseline.
        """
        language: list[str] = []

        for action in actions:
            language.append(action.action_name)

            if isinstance(action, ExtendedTeachDriverAction) and action.utterance:
                prefixed_utterance = f"<<follower>> {action.utterance}"
                language.append(prefixed_utterance)

        return language
