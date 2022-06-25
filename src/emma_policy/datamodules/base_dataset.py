import dataclasses
import random
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar, Union

import torch
from emma_datasets.datamodels import MediaType
from emma_datasets.db import DatasetDb
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem, EmmaVisualFeatures
from emma_policy.datamodules.pretrain_instances import TASK_TEMPLATES_MAP, Task
from emma_policy.utils.boxes import Boxes, pairwise_iou


DatasetReturn_Co = TypeVar(
    "DatasetReturn_Co", EmmaDatasetItem, Optional[EmmaDatasetItem], covariant=True
)


def apply_frame_shuffling(
    feature_dicts: list[dict[str, torch.Tensor]],
    fom_probability: float = 0.4,
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    """Applies frame shuffling."""
    shuffled_indices = torch.bernoulli(torch.full((len(feature_dicts),), fom_probability)).long()

    # Ensure at least two frames swap orders
    if shuffled_indices.sum() < 2:
        shuffled_indices[torch.randperm(shuffled_indices.shape[-1])[:2]] = 1

    # Get and shuffle the positions of the shuffled indices
    shuffled_pos = torch.where(shuffled_indices == 1)[0]
    target_shuffled_pos = shuffled_pos[torch.randperm(shuffled_pos.shape[-1])]
    # Make sure the frames are shuffled
    while torch.all(target_shuffled_pos == shuffled_pos):
        target_shuffled_pos = shuffled_pos[torch.randperm(shuffled_pos.shape[-1])]

    # Get the final frame order
    original_frame_order = torch.arange(len(feature_dicts))
    shuffled_pos_idx = 0
    for idx, is_shuffled in enumerate(shuffled_indices.tolist()):
        if is_shuffled:
            original_frame_order[idx] = target_shuffled_pos[shuffled_pos_idx]
            shuffled_pos_idx += 1

    feature_dicts = [feature_dicts[j] for j in original_frame_order.tolist()]
    return feature_dicts, original_frame_order


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
        bbox_match_threshold: float = 0.5,
        shuffle_frames_perc: float = 0.3,
    ) -> None:
        self.db = DatasetDb(dataset_db_path)

        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.bbox_match_threshold = bbox_match_threshold
        self.shuffle_frames_prec = shuffle_frames_perc

    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return len(self.db)

    def __getitem__(self, index: int) -> DatasetReturn_Co:
        """Get a single instance from the dataset."""
        raise NotImplementedError

    def _load_visual_features(
        self,
        features_path: Union[Path, list[Path]],
        modality: MediaType,
        truncation_side: Literal["left", "right"] = "left",
        start_offset: int = 0,
        shuffle_frames: bool = False,
        allow_empty: bool = False,
    ) -> EmmaVisualFeatures:
        """Get all the visual features from the given instance."""
        feature_dicts = self._load_feature_dicts(
            features_path=features_path,
            modality=modality,
            truncation_side=truncation_side,
            allow_empty=allow_empty,
        )
        if not len(feature_dicts) and allow_empty:
            return EmmaVisualFeatures(
                **{field.name: torch.empty(0) for field in dataclasses.fields(EmmaVisualFeatures)},
            )
        return self._prepare_emma_visual_features(
            feature_dicts=feature_dicts, start_offset=start_offset, shuffle_frames=shuffle_frames
        )

    def _prepare_emma_visual_features(  # noqa: WPS210
        self,
        feature_dicts: list[dict[str, torch.Tensor]],
        start_offset: int = 0,
        shuffle_frames: bool = False,
    ) -> EmmaVisualFeatures:
        """Prepare an EmmaVisualFeatures object."""
        object_features = []
        object_classes = []
        object_coordinates = []
        vis_tokens = []
        obj_frame_tokens = []
        object_attention_mask = []
        scene_features = []
        scene_frame_tokens = []

        num_features = len(feature_dicts)
        original_frame_order = torch.arange(num_features)
        # shuffling
        if shuffle_frames:
            feature_dicts, original_frame_order = apply_frame_shuffling(
                feature_dicts=feature_dicts, fom_probability=self.shuffle_frames_prec
            )

        for frame_idx, feature_dict in enumerate(feature_dicts):
            object_features.append(feature_dict["bbox_features"])
            object_classes.append(
                torch.tensor([torch.argmax(proba, -1) for proba in feature_dict["bbox_probas"]])
            )
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
            object_classes=torch.cat(object_classes),
            object_features=torch.cat(object_features),
            object_frame_tokens=torch.cat(obj_frame_tokens),
            scene_attention_mask=scene_attention_mask,
            scene_coordinates=scene_coordinates,
            scene_features=torch.cat(scene_features),
            scene_frame_tokens=torch.tensor(scene_frame_tokens),
            visual_token_ids=torch.cat(vis_tokens),
            original_frame_order=original_frame_order,
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
        features_path: Union[Path, list[Path]],
        modality: MediaType,
        truncation_side: Literal["left", "right"] = "left",
        allow_empty: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        """Load the visual features from file(s) and truncate them to max_frames."""
        if isinstance(features_path, Path):
            feature_dicts = self._load_feature_dicts_from_path(
                features_path=features_path, modality=modality
            )
        else:
            feature_dicts = self._load_feature_dicts_from_path_list(
                features_path=features_path, modality=modality
            )

        if not feature_dicts and not allow_empty:
            raise AssertionError("No dict of features have been loaded.")

        if self.max_frames:
            feature_dicts = self._truncate_frames(feature_dicts, truncation_side=truncation_side)
        return feature_dicts

    def _load_feature_dicts_from_path(
        self, features_path: Path, modality: MediaType
    ) -> list[dict[str, torch.Tensor]]:
        """Load the visual features from a single file."""
        if not features_path.exists():
            raise AssertionError("Provided features path does not exist.")

        if modality == MediaType.video:
            feature_dicts = [
                feature_dict["features"] for feature_dict in torch.load(features_path)["frames"]
            ]
        elif modality == MediaType.image:
            feature_dicts = [torch.load(features_path)]
        return feature_dicts

    def _load_feature_dicts_from_path_list(
        self, features_path: list[Path], modality: MediaType
    ) -> list[dict[str, torch.Tensor]]:
        """Load the visual features from a list of files."""
        feature_dicts = []
        for fpath in features_path:
            feature_dicts.extend(self._load_feature_dicts_from_path(fpath, modality))

        return feature_dicts

    def _concat_visual_features(
        self,
        visual_features_list: list[EmmaVisualFeatures],
    ) -> EmmaVisualFeatures:
        """Concatenate a list of visual features loaded from different files."""
        concat_features = {
            field.name: torch.cat(
                [getattr(visual_features, field.name) for visual_features in visual_features_list],
            ).type(getattr(visual_features_list[-1], field.name).dtype)
            for field in dataclasses.fields(EmmaVisualFeatures)
        }
        return EmmaVisualFeatures(**concat_features)

    def _convert_trajectory_to_text(
        self,
        actions: list[Any],
        feature_dicts: list[dict[str, Any]],
        visual_features: EmmaVisualFeatures,
        truncation_side: Literal["left", "right"],
    ) -> str:
        """Convert an action trajectory from an instance to a text representation."""
        raise NotImplementedError

    def _refine_instruction_text(self, raw_instruction_text: str) -> Optional[str]:
        """Makes sure that each instruction doesn't end with a fullstop."""
        if raw_instruction_text.endswith("."):
            refined_text = raw_instruction_text.replace(".", "")
        else:
            refined_text = raw_instruction_text

        return refined_text
