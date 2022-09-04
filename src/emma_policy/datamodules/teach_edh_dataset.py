import json
from pathlib import Path
from typing import Any, Literal, Union

import torch
from emma_datasets.datamodels.datasets.teach import (
    ExtendedTeachDriverAction,
    TeachDriverAction,
    TeachEdhInstance,
)
from overrides import overrides
from transformers import PreTrainedTokenizer

from emma_policy.common.settings import Settings
from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem, EmmaVisualFeatures
from emma_policy.datamodules.pretrain_dataset import split_action_name
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.utils import get_logger


logger = get_logger(__name__)

BBOX_DIAMETER = 10
BBOX_RADIUS = BBOX_DIAMETER / 2
AI2THOR_CLASS_DICT_FILE = Settings().paths.constants.joinpath("ai2thor_labels.json")


class TeachEdhDataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for EDH instances from TEACh.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

    def __init__(
        self, dataset_db_path: Path, tokenizer: PreTrainedTokenizer, max_frames: int = 0
    ) -> None:
        super().__init__(
            dataset_db_path=dataset_db_path, tokenizer=tokenizer, max_frames=max_frames
        )

        with open(AI2THOR_CLASS_DICT_FILE) as in_file:
            self.ai2thor_label_mapping = json.load(in_file)

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
        visual_features, scene_temporal_ids, object_temporal_ids = self._prepare_visual_input(
            instance
        )
        input_encoding = self.tokenizer(
            self._get_input_text_from_instance(instance, visual_features),
            return_tensors=self._return_tensor_type,
            truncation=True,
        )

        target_encoding = self.tokenizer(
            self._get_target_text_from_instance(instance, visual_features),
            return_tensors=self._return_tensor_type,
        )

        target_temporal_ids = self._make_target_temporal_ids(target_encoding.input_ids.squeeze(0))
        return EmmaDatasetItem(
            # Language
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            target_temporal_ids=target_temporal_ids,
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

    def _get_input_text_from_instance(
        self, instance: TeachEdhInstance, visual_features: EmmaVisualFeatures
    ) -> str:
        """Get the input text from a TEACh EDH instance."""
        input_text = self._get_concatenated_dialog_history(instance)

        actions = self._convert_trajectory_to_text(
            actions=instance.extended_driver_action_history,
            feature_dicts=self._load_feature_dicts(
                instance.features_path, instance.modality, allow_empty=True
            ),
            visual_features=visual_features,
            truncation_side="left",  # keep most recent actions
        )

        if actions:
            input_text = "{input_text} {sep_token} {action_trajectory}".format(
                input_text=input_text,
                sep_token=self.tokenizer.sep_token,
                action_trajectory=actions,
            )

        #  Add action execution task prefix
        input_text = self._get_random_template_for_task(Task.action_execution).format(
            instruction=input_text,
        )
        return input_text

    def _get_target_text_from_instance(
        self, instance: TeachEdhInstance, visual_features: EmmaVisualFeatures
    ) -> str:
        """Get the target text from a TEACh EDH instance."""
        return self._convert_trajectory_to_text(
            actions=instance.driver_actions_future,
            feature_dicts=self._load_feature_dicts(
                instance.future_features_path, instance.modality, allow_empty=True
            ),
            visual_features=visual_features,
            truncation_side="right",  # keep first actions
        )

    def _get_concatenated_dialog_history(
        self, instance: TeachEdhInstance, cleaned: bool = True
    ) -> str:
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
        concat_dialog_history[-1] = self._refine_instruction_text(concat_dialog_history[-1])  # type: ignore[assignment]
        return " ".join(concat_dialog_history)

    def _convert_trajectory_to_text(
        self,
        actions: Union[list[ExtendedTeachDriverAction], list[TeachDriverAction]],
        feature_dicts: list[dict[str, Any]],
        visual_features: EmmaVisualFeatures,
        truncation_side: Literal["left", "right"] = "left",
    ) -> str:
        """Convert a list of driver actions to a single string."""
        if self.max_frames:
            feature_dicts = self._truncate_frames(feature_dicts, truncation_side=truncation_side)
            actions = self._truncate_frames(actions, truncation_side=truncation_side)

        trajectory = []

        for action_idx, action in enumerate(actions):
            trajectory.extend(split_action_name(action.action_name))

            if action.obj_interaction_action == 1:
                ground_truth_centroid_coord = (
                    action.x * feature_dicts[action_idx]["width"],
                    action.y * feature_dicts[action_idx]["height"],
                )
                ground_truth_bbox = torch.tensor(
                    [
                        ground_truth_centroid_coord[0] - BBOX_RADIUS,  # x1
                        ground_truth_centroid_coord[1] - BBOX_RADIUS,  # y1
                        ground_truth_centroid_coord[0] + BBOX_RADIUS,  # x2
                        ground_truth_centroid_coord[1] + BBOX_RADIUS,  # y2
                    ]
                )
                # normalized coordinates
                ground_truth_bbox[[0, 2]] /= feature_dicts[action_idx]["width"]
                ground_truth_bbox[[1, 3]] /= feature_dicts[action_idx]["height"]

                # Get the index of the objects from the current frame. Frames start from 1.
                frame_token = self.tokenizer.convert_tokens_to_ids(f"<frame_token_{action_idx+1}>")
                frame_objects = visual_features.object_frame_tokens == frame_token

                matched_index, gt_flags = self._best_match_features(
                    ground_truth_bbox=ground_truth_bbox.unsqueeze(0),
                    object_coordinates_bbox=visual_features.object_coordinates[frame_objects],
                    threshold=0,  # this is set to 0 to filter out boxes not matching at all
                )

                # we first add the class of the object we want to interact with
                trajectory.append(action.object_name.lower())

                # then if we have a matching bounding box, we add the visual token as well
                found_matched_object = gt_flags[0]
                if found_matched_object:
                    trajectory.append(
                        self.tokenizer.decode(
                            visual_features.visual_token_ids[frame_objects][matched_index[0]]
                        )
                    )

            trajectory[-1] = f"{trajectory[-1]}{self.tokenizer.sep_token}"

        return " ".join(trajectory)

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

    def _make_target_temporal_ids(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """Get future indices for target tokens."""
        target_temporal_ids = torch.zeros_like(target_tokens)
        separator_indices = torch.where(target_tokens == self.tokenizer.sep_token_id)[0]
        target_temporal_ids[separator_indices + 1] = 1
        # Increment the frame id after each observed separator token
        target_temporal_ids = torch.cumsum(target_temporal_ids, -1) + 1
        return target_temporal_ids

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
