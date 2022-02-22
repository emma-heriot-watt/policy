from dataclasses import dataclass

import torch


@dataclass
class EmmaVisualFeatures:
    """Set of visual features for a training instance."""

    scene_features: torch.Tensor
    scene_coordinates: torch.Tensor
    object_features: torch.Tensor
    object_coordinates: torch.Tensor
    visual_token_ids: torch.Tensor
    scene_frame_ids: torch.Tensor
    object_frame_ids: torch.Tensor
    scene_attention_mask: torch.Tensor
    object_attention_mask: torch.Tensor


@dataclass
class EmmaDatasetItem:
    """Output for the dataset reader."""

    input_token_ids: torch.Tensor
    text_attention_mask: torch.Tensor
    target_token_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    object_attention_mask: torch.Tensor
    object_coordinates: torch.Tensor
    object_features: torch.Tensor
    object_frame_ids: torch.Tensor
    scene_attention_mask: torch.Tensor
    scene_coordinates: torch.Tensor
    scene_features: torch.Tensor
    scene_frame_ids: torch.Tensor
    visual_token_ids: torch.Tensor


@dataclass
class EmmaDatasetPadding:
    """Output for the dataset reader."""

    attention_mask: int = 0
    global_attention_mask: int = 0
    input_token_ids: int = 1
    text_attention_mask: int = 0
    target_token_ids: int = -100
    decoder_attention_mask: int = 0
    object_attention_mask: int = 0
    object_coordinates: int = 0
    object_features: int = 0
    object_frame_ids: int = 0
    scene_attention_mask: int = 0
    scene_coordinates: int = 0
    scene_features: int = 0
    scene_frame_ids: int = 0
    visual_token_ids: int = 1


@dataclass
class EmmaDatasetBatch:
    """Output for the dataset reader."""

    attention_mask: torch.Tensor
    global_attention_mask: torch.Tensor
    input_token_ids: torch.Tensor
    text_attention_mask: torch.Tensor
    target_token_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    object_attention_mask: torch.Tensor
    object_coordinates: torch.Tensor
    object_features: torch.Tensor
    object_frame_ids: torch.Tensor
    scene_attention_mask: torch.Tensor
    scene_coordinates: torch.Tensor
    scene_features: torch.Tensor
    scene_frame_ids: torch.Tensor
    visual_token_ids: torch.Tensor
