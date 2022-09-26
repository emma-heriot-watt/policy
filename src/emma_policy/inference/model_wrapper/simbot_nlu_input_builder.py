import logging
from typing import Any

import torch
from pytorch_lightning.utilities import move_data_to_device
from transformers import BatchEncoding, PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import prepare_emma_visual_features
from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetBatch,
    EmmaDatasetItem,
    EmmaVisualFeatures,
)
from emma_policy.inference.api.simbot_state import GenerateRequest


logger = logging.getLogger(__name__)

FeatureDictsType = list[dict[str, Any]]


class SimBotNLUInputBuilder:
    """Build the input for the Emma NLU model."""

    def __init__(self, tokenizer: PreTrainedTokenizer, device: str = "cpu") -> None:
        self._tokenizer = tokenizer
        self.device = device

    def __call__(self, request: GenerateRequest) -> EmmaDatasetBatch:
        """Process the environment output into a batch for the model.

        The sample batch provides the set of previous observations and previous actions taken by
        the agent in the environment.
        """
        instruction = request.dialogue_history[-1].utterance
        logger.debug(f"Preparing NLU input for instruction: {instruction}")
        encoded_inputs = self._prepare_input_text(instruction)
        feature_dicts = self._prepare_feature_dicts(request.environment_history[-1].features)
        visual_features = self._prepare_visual_features(feature_dicts)
        dataset_item = self._create_emma_dataset_item(
            visual_features=visual_features, encoded_inputs=encoded_inputs
        )
        return self._create_emma_dataset_batch(dataset_item)

    def _prepare_input_text(self, instruction: str) -> BatchEncoding:
        source_text = f"Predict the system act: {instruction}"
        return self._tokenizer.encode_plus(source_text, return_tensors="pt", truncation=True)

    def _prepare_feature_dicts(self, feature_dicts: FeatureDictsType) -> FeatureDictsType:
        """Convert feature dicts to tensors."""
        feature_dicts_tensors = []
        for feature_dict in feature_dicts:
            feature_dicts_tensors.append(
                {
                    name: self._convert_to_tensor(instance)
                    for name, instance in feature_dict.items()
                }
            )
        return feature_dicts_tensors

    def _convert_to_tensor(self, instance: Any) -> Any:
        """Convert an element of a feature dict from list to tensor."""
        if isinstance(instance, list):
            instance = torch.tensor(instance)
        return instance

    def _prepare_visual_features(self, feature_dicts: FeatureDictsType) -> EmmaVisualFeatures:
        visual_features = prepare_emma_visual_features(
            feature_dicts=feature_dicts,
            tokenizer=self._tokenizer,
        )
        return visual_features

    def _create_emma_dataset_item(
        self,
        visual_features: EmmaVisualFeatures,
        encoded_inputs: BatchEncoding,
    ) -> EmmaDatasetItem:
        """Create the `EmmaDatasetItem` for a given set of observations and actions."""
        return EmmaDatasetItem(
            # Text
            input_token_ids=encoded_inputs.input_ids.squeeze(0),
            text_attention_mask=encoded_inputs.attention_mask.squeeze(0),
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
        )

    def _create_emma_dataset_batch(
        self,
        dataset_item: EmmaDatasetItem,
    ) -> EmmaDatasetBatch:
        """Create the `EmmaDatasetBatch` for a given set of observations and actions."""
        batch = collate_fn([dataset_item])

        return move_data_to_device(batch, self.device)
