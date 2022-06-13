import logging
from dataclasses import asdict
from typing import Any, Optional

import torch
from emma_datasets.datamodels.datasets import TeachEdhInstance
from overrides import overrides
from PIL.Image import Image
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem, EmmaVisualFeatures
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.datamodules.teach_edh_dataset import TeachEdhDataset
from emma_policy.inference.api.settings import FeatureExtractorSettings
from emma_policy.inference.model_wrapper.feature_client import FeatureClient


logger = logging.getLogger(__name__)


class TeachEdhInferenceDataset(TeachEdhDataset):
    """TeachEdh Dataset for inference."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        feature_extractor_settings: FeatureExtractorSettings,
        max_frames: int = 100,
    ) -> None:
        # This is what is expected by the `TeachEdhDataset`
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.previous_frame: Optional[Image] = None

        self.client = FeatureClient(feature_extractor_settings=feature_extractor_settings)

        self._trajectory_visual_features: list[EmmaVisualFeatures] = []
        self._history_visual_features: EmmaVisualFeatures
        self._original_history_length: int = -1
        self._feature_dicts: list[dict[str, Any]] = []
        self._input_encoding: BatchEncoding
        self._current_bbox_probas: Optional[torch.Tensor]
        self._current_coordinates: Optional[torch.Tensor]

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        feature_extractor_settings: FeatureExtractorSettings,
        max_frames: int = 0,
        max_lang_tokens: Optional[int] = None,
    ) -> "TeachEdhInferenceDataset":
        """Instantiate TeachEdhInferenceDataset."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if max_lang_tokens:
            tokenizer.model_max_length = max_lang_tokens

        return cls(
            tokenizer=tokenizer,
            max_frames=max_frames,
            feature_extractor_settings=feature_extractor_settings,
        )

    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return 1

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> None:
        """Get the single instance during inference."""
        raise NotImplementedError("Dont call __getitem__")

    def start_new_edh_instance(
        self,
        edh_instance: TeachEdhInstance,
        edh_history_images: list[Image],
        edh_name: Optional[str] = None,
    ) -> bool:
        """Clear the state and start a new EDH instance."""
        logger.debug(f"Preparing visual features for `{edh_instance.instance_id}`")

        self._history_visual_features = self.prepare_visual_features(edh_history_images)
        self._trajectory_visual_features = []
        self._original_history_length = min(self.max_frames, len(edh_history_images))
        self._feature_dicts = [
            {"width": image.size[0], "height": image.size[1]} for image in edh_history_images
        ]

        logger.debug(f"Tokenizing input text `{edh_instance.instance_id}`")
        self._input_encoding = self.tokenizer(
            self._get_input_text_from_instance(edh_instance, self._history_visual_features),
            return_tensors=self._return_tensor_type,
            truncation=True,
        )
        self._current_bbox_probas = None
        self._current_coordinates = None
        self.previous_frame = edh_history_images[-1]

        logger.debug(f"Model prepared `{edh_instance.instance_id}`")
        return True

    def get_next_dataset_instance(self, current_frame: Image) -> EmmaDatasetItem:
        """Get the emma input given the current egocentric view."""
        return self._convert_instance_to_emma_dataset_item(current_frame)

    def get_current_object_probas(self) -> torch.Tensor:
        """Return the bounding box probabilities from the current egocentric view."""
        if self._current_bbox_probas is None:
            raise AssertionError(
                "Do not try to get current object probabilities before calling `get_next_dataset_instance`"
            )

        return self._current_bbox_probas

    def get_current_coordinates(self) -> torch.Tensor:  # noqa: WPS615
        """Return the bbox coordinates from the current egocentric view."""
        if self._current_coordinates is None:
            raise AssertionError(
                "Do not try to get current object probabilities before calling `get_next_dataset_instance`"
            )
        return self._current_coordinates

    def prepare_visual_features(
        self, edh_history_images: list[Image], start_offset: int = 0
    ) -> EmmaVisualFeatures:
        """Prepare an EmmaVisualFeatures object."""
        if self.max_frames:
            edh_history_images = edh_history_images[-self.max_frames :]

        # TODO: make this work in batches
        logger.debug("Building the feature dicts")
        feature_dicts: list[dict[str, Any]] = []

        for idx, edh_history_image in enumerate(edh_history_images):
            logger.debug(f"Requesting features for image {idx}/{len(edh_history_images)}")

            feature_response = self.client.post_request(edh_history_image)
            feature_dicts.append(asdict(feature_response))

        self._current_bbox_probas = feature_dicts[-1]["bbox_probas"]
        self._current_coordinates = feature_dicts[-1]["bbox_coords"]

        logging.debug("Converting feature dicts to `EmmaVisualFeatures` object")
        return self._prepare_emma_visual_features(
            feature_dicts=feature_dicts, start_offset=start_offset
        )

    def _prepare_visual_input(
        self, current_frame: Image
    ) -> tuple[EmmaVisualFeatures, torch.Tensor, torch.Tensor]:
        """Load history and future visual features and compute temporal ids."""
        offset = self._original_history_length + len(self._trajectory_visual_features)
        # Update the features seen in the trajectory
        self._trajectory_visual_features.append(
            self.prepare_visual_features(edh_history_images=[current_frame], start_offset=offset)
        )
        self._trajectory_visual_features = self._truncate_frames(
            self._trajectory_visual_features, truncation_side="left"
        )
        # Fix frame tokens after truncation
        for idx, frame_features in enumerate(self._trajectory_visual_features):
            new_frame_token = self.tokenizer.convert_tokens_to_ids(
                f"<frame_token_{idx+self._original_history_length+1}>"
            )
            self._trajectory_visual_features[idx].scene_frame_tokens = torch.tensor(
                [new_frame_token]
            )
            self._trajectory_visual_features[idx].object_frame_tokens = torch.tensor(
                [new_frame_token] * frame_features.object_frame_tokens.shape[0],  # noqa: WPS435
            )

        # Concatenate history and trajectory tokens
        visual_features_list = [self._history_visual_features] + self._trajectory_visual_features
        visual_features = self._concat_visual_features(visual_features_list)

        scene_temporal_ids, object_temporal_ids = self._make_image_temporal_ids(
            feature_len_history=self._original_history_length,
            feature_len_future=len(self._trajectory_visual_features),
            object_frame_tokens=visual_features.object_frame_tokens,
        )
        return visual_features, scene_temporal_ids, object_temporal_ids

    def _convert_instance_to_emma_dataset_item(self, current_frame: Image) -> EmmaDatasetItem:
        """Convert the EDH instance to an instance of `EmmaDatasetItem`."""
        visual_features, scene_temporal_ids, object_temporal_ids = self._prepare_visual_input(
            current_frame=current_frame
        )

        return EmmaDatasetItem(
            # Language
            input_token_ids=self._input_encoding.input_ids.squeeze(0),
            text_attention_mask=self._input_encoding.attention_mask.squeeze(0),
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
            feature_dicts=self._feature_dicts,
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
