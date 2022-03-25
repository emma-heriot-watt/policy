import dataclasses
import random
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar

import torch
from emma_datasets.datamodels import MediaType
from emma_datasets.db import DatasetDb
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem, EmmaVisualFeatures
from emma_policy.datamodules.pretrain_instances import TASK_TEMPLATES_MAP, Task
from emma_policy.utils.boxes import Boxes, pairwise_iou


DatasetReturn_Co = TypeVar(
    "DatasetReturn_Co",
    EmmaDatasetItem,
    Optional[EmmaDatasetItem],
    covariant=True,
)


class EmmaBaseDataset(Dataset[DatasetReturn_Co]):
    """Common torch Dataset for easily getting dataset from DatasetDb files for modelling.

    The `__getitem__` needs implementing in subclasses.

    A type is still needed when inheriting from this, but only `EmmaDatasetItem` or
    `Optional[EmmaDatasetItem]` are allowed. For example, you could do one of the following:

        1.  class MyNewDataset(EmmaBaseDataset[EmmaDatasetItem]):
                ...

        2.  class MyOtherNewDataset(EmmaBaseDataset[Optional[EmmaDatasetItem]]):
                ...
    """

    _return_tensor_type: str = "pt"

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_frames: int = 0,
    ) -> None:
        self.db = DatasetDb(dataset_db_path)

        self.tokenizer = tokenizer
        self.max_frames = max_frames

    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return len(self.db)

    def __getitem__(self, index: int) -> DatasetReturn_Co:
        """Get a single instance from the dataset."""
        raise NotImplementedError

    def _load_visual_features(
        self,
        features_path: Path,
        modality: MediaType,
        truncation_side: Literal["left", "right"] = "left",
        start_offset: int = 0,
    ) -> EmmaVisualFeatures:
        """Get all the visual features from the given instance."""
        feature_dicts = self._load_feature_dicts(
            features_path=features_path,
            modality=modality,
            truncation_side=truncation_side,
        )

        object_features = []
        object_coordinates = []
        vis_tokens = []
        obj_frame_tokens = []
        object_attention_mask = []
        scene_features = []
        scene_frame_tokens = []

        for frame_idx, feature_dict in enumerate(feature_dicts):
            object_features.append(feature_dict["bbox_features"])
            image_coords = feature_dict["bbox_coords"]

            # normalized coordinates
            image_coords[:, (0, 2)] /= feature_dict["width"]
            image_coords[:, (1, 3)] /= feature_dict["height"]
            object_coordinates.append(image_coords)

            scene_features.append(feature_dict["cnn_features"].unsqueeze(0))

            feature_count = object_features[-1].shape[0]

            curr_vis_tokens = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(
                    [f"<vis_token_{idx+1}>" for idx in range(feature_count)]
                ),
                dtype=torch.long,
            )
            vis_tokens.append(curr_vis_tokens)

            frame_token = self.tokenizer.convert_tokens_to_ids(
                f"<frame_token_{frame_idx+start_offset+1}>"
            )
            obj_frame_tokens.append(
                curr_vis_tokens.new_full(
                    curr_vis_tokens.shape,
                    fill_value=frame_token,  # type: ignore[arg-type]
                )
            )
            scene_frame_tokens.append(frame_token)
            object_attention_mask.append(torch.ones_like(curr_vis_tokens, dtype=torch.bool))

        num_frames = len(scene_features)
        scene_attention_mask = torch.ones(num_frames, dtype=torch.bool)
        scene_coordinates = torch.tensor([0, 0, 1.0, 1.0]).repeat(num_frames, 1)

        emma_visual_features = EmmaVisualFeatures(
            object_attention_mask=torch.cat(object_attention_mask),
            object_coordinates=torch.cat(object_coordinates),
            object_features=torch.cat(object_features),
            object_frame_tokens=torch.cat(obj_frame_tokens),
            scene_attention_mask=scene_attention_mask,
            scene_coordinates=scene_coordinates,
            scene_features=torch.cat(scene_features),
            scene_frame_tokens=torch.tensor(scene_frame_tokens),
            visual_token_ids=torch.cat(vis_tokens),
        )

        return emma_visual_features

    def _best_match_features(
        self,
        ground_truth_bbox: torch.Tensor,
        object_coordinates_bbox: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the predicted bounding box with the highest IoU (best feature vector)."""
        ground_truth_box = Boxes(ground_truth_bbox)
        estimated_box = Boxes(object_coordinates_bbox)
        ious = pairwise_iou(ground_truth_box, estimated_box)

        # ignore all the ious below threshold
        # ious[ious < threshold] = 0
        matched_values, matched_indices = torch.max(ious, dim=1)

        # keep track whether ground truth bbox is mapped to a feature
        ground_truth_flags = matched_values > threshold

        return matched_indices, ground_truth_flags

    def _get_task_as_tensor(self, task: Task) -> torch.Tensor:
        """Convert the given task to a Tensor."""
        return torch.tensor([Task.get_index(task)], dtype=torch.long)

    def _get_random_template_for_task(self, task: Task) -> str:
        """Choose a random template for the given task."""
        return random.choice(TASK_TEMPLATES_MAP[task])

    def _truncate_frames(
        self, frame_sequence: list[Any], truncation_side: Literal["left", "right"] = "left"
    ) -> list[Any]:
        """Truncate a list where each element corresponds to a frame.

        The list elements can be either feature_dicts or actions of TEACh EDH instances.
        """
        if truncation_side == "left":
            frame_sequence = frame_sequence[-self.max_frames :]
        else:
            frame_sequence = frame_sequence[: self.max_frames]
        return frame_sequence

    def _load_feature_dicts(
        self,
        features_path: Path,
        modality: MediaType,
        truncation_side: Literal["left", "right"] = "left",
    ) -> list[dict[str, torch.Tensor]]:
        """Load the visual features from file and truncate them to max_frames."""
        if not features_path.exists():
            raise AssertionError("Provided features path does not exist.")

        if modality == MediaType.video:
            feature_dicts = [
                feature_dict["features"] for feature_dict in torch.load(features_path)["frames"]
            ]

        elif modality == MediaType.image:
            feature_dicts = [torch.load(features_path)]

        if not feature_dicts:
            raise AssertionError("No dict of features have been loaded.")

        if self.max_frames:
            feature_dicts = self._truncate_frames(feature_dicts, truncation_side=truncation_side)
        return feature_dicts

    def _concat_visual_features(
        self,
        visual_features_list: list[EmmaVisualFeatures],
    ) -> EmmaVisualFeatures:
        """Concatenate a list of visual features loaded from different files."""
        concat_features = {
            field.name: torch.cat(
                [getattr(visual_features, field.name) for visual_features in visual_features_list],
            )
            for field in dataclasses.fields(EmmaVisualFeatures)
        }
        return EmmaVisualFeatures(**concat_features)
