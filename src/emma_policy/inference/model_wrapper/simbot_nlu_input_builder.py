import logging
import re
from typing import Any, Optional

import torch
from emma_common.datamodels import EmmaPolicyRequest
from pytorch_lightning.utilities import move_data_to_device
from transformers import BatchEncoding, PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import prepare_emma_visual_features
from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetBatch,
    EmmaDatasetItem,
    EmmaVisualFeatures,
)
from emma_policy.utils.datamodels.simbot import EMPTY_INVENTORY, format_instruction


logger = logging.getLogger(__name__)

FeatureDictsType = list[dict[str, Any]]


class SimBotNLUInputBuilder:
    """Build the input for the Emma NLU model."""

    def __init__(self, tokenizer: PreTrainedTokenizer, device: str = "cpu") -> None:
        self._tokenizer = tokenizer
        self._device = device

    def __call__(self, request: EmmaPolicyRequest) -> tuple[EmmaDatasetBatch, str]:
        """Process the environment output into a batch for the model.

        The sample batch provides the set of previous observations and previous actions taken by
        the agent in the environment.
        """
        # Add a fullstop at the end and lowercase
        inventory = EMPTY_INVENTORY if request.inventory is None else request.inventory
        instruction = f"Inventory: {inventory}. {request.dialogue_history[-1].utterance}"
        instruction = format_instruction(instruction)
        logger.debug(f"Preparing NLU input for instruction: {instruction}")
        encoded_inputs = self._prepare_input_text(instruction)
        feature_dicts = [feature.dict() for feature in request.environment_history[-1].features]
        feature_dicts = self._prepare_feature_dicts(feature_dicts)
        visual_features = self._prepare_visual_features(feature_dicts)
        dataset_item = self._create_emma_dataset_item(
            visual_features=visual_features, encoded_inputs=encoded_inputs
        )
        return self._create_emma_dataset_batch(dataset_item), instruction

    def check_sticky_note_case(
        self, request: EmmaPolicyRequest, default_prediction: str
    ) -> Optional[str]:
        """Check if the instruction refers to a sticky note."""
        features = request.environment_history[-1].features

        entity_labels = features[0].entity_labels

        ignore_instruction = any(
            [
                len(request.environment_history) > 1,
                len(features) > 1,
                entity_labels is None,
                len(request.dialogue_history) > 1,
                request.dialogue_history[-1].role == "agent",
            ]
        )
        if ignore_instruction:
            return None

        if "Sticky Note" in entity_labels:
            patterns = "|".join(["sticky", "clue", "hint", "postit", "posted"])
            search_pattern = f"({patterns})"
            search_result = re.search(search_pattern, request.dialogue_history[-1].utterance)

            if search_result is not None:
                return default_prediction
        else:
            return "<act><no_match> sticky note"

        return None

    def _prepare_input_text(self, instruction: str) -> BatchEncoding:
        source_text = f"Predict the system act: {instruction}"
        return self._tokenizer.encode_plus(source_text, return_tensors="pt", truncation=True)

    def _prepare_feature_dicts(self, feature_dicts: FeatureDictsType) -> FeatureDictsType:
        """Convert feature dicts to tensors."""
        feature_dicts_tensors = []
        for feature_dict in feature_dicts:
            for name, instance in feature_dict.items():
                if name not in {"class_labels", "entity_labels"}:
                    feature_dict[name] = self._convert_to_tensor(instance)
            feature_dicts_tensors.append(feature_dict)
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

        prev_device = batch.input_token_ids.device
        batch = move_data_to_device(batch, self._device)
        logger.debug(f"Moved batch from {prev_device} to {batch.input_token_ids.device}")
        return batch
