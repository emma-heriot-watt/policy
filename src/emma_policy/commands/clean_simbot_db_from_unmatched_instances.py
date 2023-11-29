import argparse
import random
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from emma_datasets.constants.simbot.simbot import get_arena_definitions
from emma_datasets.datamodels import DatasetName
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_label_from_object_id,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    SimBotClarificationTypes,
    SimBotInstructionInstance,
    SimBotObjectAttributes,
)
from emma_datasets.db import DatasetDb
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

from emma_policy.datamodules.base_dataset import best_match_features
from emma_policy.datamodules.emma_dataclasses import EmmaVisualFeatures
from emma_policy.datamodules.simbot_action_datamodule import prepare_action_tokenizer
from emma_policy.datamodules.simbot_action_dataset import (
    SimBotActionDataset,
    compressed_mask_is_bbox,
)
from emma_policy.datamodules.simbot_cr_dataset import (
    SimBotCRDataset,
    action_is_object_interaction,
)
from emma_policy.utils import decompress_simbot_mask, get_logger


logger = get_logger(__name__)


class FilterSimBotDB:
    """Filter the DB instances based on the match with predicted bounding boxes.

    Discard instances where the groundtruth bounding box cannot be matched to a predicted one.
    """

    def __init__(
        self,
        train_input_db_file: Path,
        valid_input_db_file: Path,
        train_output_db_file: Path,
        valid_output_db_file: Path,
        iou_threshold: float = 0.5,
        minimum_iou_threshold: float = 0.1,
        simbot_db_type: Literal["action", "cr"] = "action",
        matching_strategy: Literal["threshold_only", "threshold_and_label"] = "threshold_only",
        model_name: str = "heriot-watt/emma-base",
    ) -> None:
        self._train_input_db_file = train_input_db_file
        self._valid_input_db_file = valid_input_db_file
        self._train_output_db_file = train_output_db_file
        self._valid_output_db_file = valid_output_db_file
        self._iou_threshold = iou_threshold
        self._minimum_iou_threshold = minimum_iou_threshold
        tokenizer = prepare_action_tokenizer(
            model_name=model_name, tokenizer_truncation_side="right", max_lang_tokens=None
        )

        arena_definitions = get_arena_definitions()
        self._object_assets_to_names = arena_definitions["asset_to_label"]
        self._label_to_idx = arena_definitions["label_to_idx"]

        self.dataset: Union[SimBotActionDataset, SimBotCRDataset]
        if simbot_db_type == "action":
            self._action_idx = -1
            self._purge_instance = self._discard_action_unmatched_instance
            self.dataset_name = DatasetName.simbot_actions.name
            self.dataset = SimBotActionDataset(
                dataset_db_path=valid_input_db_file,
                tokenizer=tokenizer,
                iou_threshold=args.iou_threshold,
            )
        else:
            self._action_idx = 0
            self._purge_instance = self._discard_cr_unmatched_instance
            self.dataset_name = DatasetName.simbot_clarifications.name
            self.dataset = SimBotCRDataset(
                dataset_db_path=valid_input_db_file,
                tokenizer=tokenizer,
                iou_threshold=args.iou_threshold,
            )
        if matching_strategy == "threshold_only":
            self.match_bboxes = self._match_based_on_iou_threshold
        else:
            self.match_bboxes = self._match_based_on_iou_threshold_and_label

        self._image_width = 300
        self._image_height = 300

    def run(self) -> None:
        """Filter the training and validation dbs."""
        logger.info(f"Cleaning the validation db: {self._valid_input_db_file}")
        self.run_for_split(self._valid_input_db_file, self._valid_output_db_file, "validation")
        logger.info(f"Cleaning the training db: {self._train_input_db_file}")
        self.run_for_split(self._train_input_db_file, self._train_output_db_file, "training")

    def run_for_split(
        self,
        input_db_path: Path,
        output_db_path: Path,
        dataset_split: str,
    ) -> None:
        """Filter the given db."""
        keep_instances = []
        db = DatasetDb(input_db_path)
        db_indices = list(range(len(db)))
        for idx in tqdm(db_indices):
            with db:
                instance = SimBotInstructionInstance.parse_raw(db[idx])
            if instance.actions[self._action_idx].type == "Search":
                updated_instance = self._discard_search_unmatched_objects(instance)
            else:
                updated_instance = self._purge_instance(instance)
            if updated_instance is not None:
                keep_instances.append(updated_instance)

        self._write_db(keep_instances, output_db_path, dataset_split=dataset_split)
        logger.info(f"Final db has {len(keep_instances)} out of the original {len(db)}")

    def _discard_search_unmatched_objects(
        self, instance: SimBotInstructionInstance
    ) -> Optional[SimBotInstructionInstance]:
        """Discard objects that do not match any predicted bounding box."""
        action_object_metadata = instance.actions[self._action_idx].get_action_data["object"]
        ground_truth_bboxes = action_object_metadata["mask"]
        # Negative search instances are by default clean
        # Instances that are not paraphrasable are from human annotations
        if ground_truth_bboxes is None or not instance.paraphrasable:
            return instance

        visual_features, _, _ = self._load_visual_features(instance)
        should_keep_candidate_idx = []
        for object_candidate_idx, object_id in enumerate(action_object_metadata["id"]):
            ground_truth_bbox = ground_truth_bboxes[object_candidate_idx]
            ground_truth_bbox = torch.tensor(
                ground_truth_bbox,
                dtype=torch.float32,
            ).unsqueeze(0)

            object_name = get_object_label_from_object_id(object_id, self._object_assets_to_names)

            # If there is a matching box, this object cannot be mapped to a bbox.
            if not self.match_bboxes(ground_truth_bbox, visual_features, object_name):
                continue
            should_keep_candidate_idx.append(object_candidate_idx)

        if not should_keep_candidate_idx:
            return None
        # Update the action object metadata with the valid objects.
        action_object_metadata["id"] = [
            action_object_metadata["id"][idx] for idx in should_keep_candidate_idx
        ]
        action_object_metadata["mask"] = [
            action_object_metadata["mask"][idx] for idx in should_keep_candidate_idx
        ]
        action_object_metadata["attributes"] = [
            action_object_metadata["attributes"][idx] for idx in should_keep_candidate_idx
        ]
        instance.actions[self._action_idx].search["object"].update(  # noqa: WPS219
            action_object_metadata
        )

        use_selected_object = (
            "selected_object" in instance.actions[self._action_idx].search
            and instance.actions[self._action_idx].search["selected_object"]["id"]  # noqa: WPS219
            in action_object_metadata["id"]
        )
        if use_selected_object:
            return instance
        else:
            # Resample the default instruction
            random_idx = random.choice(range(len(should_keep_candidate_idx)))
            instruction = self.dataset.low_level_paraphraser(
                action_type="search",
                object_id=action_object_metadata["id"][random_idx],
                object_attributes=SimBotObjectAttributes(
                    **action_object_metadata["attributes"][random_idx]
                ),
            )
            instance.actions[self._action_idx].search["selected_object"] = {
                "id": action_object_metadata["id"][random_idx],
                "attributes": action_object_metadata["attributes"][random_idx],
            }
            instance.instruction.instruction = instruction
        return instance

    def _discard_action_unmatched_instance(
        self, instance: SimBotInstructionInstance
    ) -> Optional[SimBotInstructionInstance]:
        """Discard instances where the target object does not match any predicted bounding box."""
        keep_instance = True
        visual_features, frames, objects_per_frame = self._load_visual_features(
            instance=instance,
            target_frames=self.dataset.get_target_frames(instance),  # type: ignore[union-attr]
        )
        # Check that all interaction actions have matched bounding boxes
        for action in instance.actions:
            if not action_is_object_interaction(action):
                continue
            action_object_metadata = action.get_action_data["object"]
            object_id = action_object_metadata["id"]
            object_name = get_object_label_from_object_id(
                object_id=object_id,
                object_assets_to_names=self._object_assets_to_names,
            )
            image_index = action_object_metadata["colorImageIndex"]
            object_name_with_tokens = self.dataset.map_object_to_visual_token(  # type: ignore[union-attr]
                object_name=object_name,
                action=action,
                image_index=image_index,
                visual_features=visual_features,
                frames=frames,
                objects_per_frame=objects_per_frame,
            )

            if "vis_token" not in object_name_with_tokens:
                keep_instance = False
        if keep_instance:
            return instance
        return None

    def _discard_cr_unmatched_instance(
        self, instance: SimBotInstructionInstance
    ) -> Optional[SimBotInstructionInstance]:
        """Discard instances where the target object does not match any predicted bounding box."""
        intance_questions = instance.instruction.necessary_question_answers
        # For location questions (no_match), it's okay to not match a bounding box
        if intance_questions and intance_questions[0] == SimBotClarificationTypes.location:
            return instance

        if not action_is_object_interaction(instance.actions[self._action_idx]):
            return instance
        action_object_metadata = instance.actions[self._action_idx].get_action_data["object"]
        object_id = action_object_metadata["id"]
        object_mask = action_object_metadata["mask"]
        if compressed_mask_is_bbox(object_mask):
            ground_truth_bbox = torch.tensor(object_mask, dtype=torch.float32).unsqueeze(0)
        else:
            gt_binary_mask = decompress_simbot_mask(object_mask)
            ground_truth_bbox = masks_to_boxes(torch.tensor(gt_binary_mask).unsqueeze(0))

        frame_idx = action_object_metadata.get("colorImageIndex", 0)
        visual_features, _, _ = self._load_visual_features(instance, frame_idx=frame_idx)

        object_name = get_object_label_from_object_id(object_id, self._object_assets_to_names)

        if not self.match_bboxes(ground_truth_bbox, visual_features, object_name):
            return None
        return instance

    def _match_based_on_iou_threshold(
        self,
        ground_truth_bbox: torch.Tensor,
        visual_features: EmmaVisualFeatures,
        object_name: str = "",
    ) -> bool:
        """Match objects that exceed a threshold IoU."""
        ground_truth_bbox[:, (0, 2)] /= self._image_width
        ground_truth_bbox[:, (1, 3)] /= self._image_height
        _, ground_truth_flags = best_match_features(
            ground_truth_bbox=ground_truth_bbox,
            object_coordinates_bbox=visual_features.object_coordinates,
            threshold=self._iou_threshold,
        )
        return ground_truth_flags[0].item()  # type: ignore[return-value]

    def _match_based_on_iou_threshold_and_label(
        self,
        ground_truth_bbox: torch.Tensor,
        visual_features: EmmaVisualFeatures,
        object_name: str = "",
    ) -> bool:
        """Match objects that exceed a threshold IoU or match in terms of label."""
        ground_truth_bbox[:, (0, 2)] /= self._image_width
        ground_truth_bbox[:, (1, 3)] /= self._image_height
        _, ground_truth_flags = best_match_features(
            ground_truth_bbox=ground_truth_bbox,
            object_coordinates_bbox=visual_features.object_coordinates,
            threshold=self._iou_threshold,
        )
        if ground_truth_flags[0].item():
            return True
        matched_indices, ground_truth_flags = best_match_features(
            ground_truth_bbox=ground_truth_bbox,
            object_coordinates_bbox=visual_features.object_coordinates,
            threshold=self._minimum_iou_threshold,
        )
        if ground_truth_flags[0].item():
            return True
        target_class = self._label_to_idx[object_name]
        return target_class == visual_features.object_classes[matched_indices[0]]

    def _load_visual_features(
        self,
        instance: SimBotInstructionInstance,
        target_frames: Optional[list[int]] = None,
        frame_idx: int = 0,
    ) -> tuple[EmmaVisualFeatures, list[str], list[int]]:
        """Load visual features depending on the dataset."""
        frames: list[str] = []
        objects_per_frame: list[int] = []
        if self.dataset_name == DatasetName.simbot_actions.name:
            if target_frames is None:
                target_frames = [0 for _ in instance.actions]
            return self.dataset._load_visual_features(  # noqa: WPS437
                features_path=instance.features_path,
                target_frames=target_frames,
            )
        else:
            visual_features = self.dataset._load_visual_features(  # noqa: WPS437
                instance.features_path[0], frame_idx=frame_idx
            )
        return visual_features, frames, objects_per_frame

    def _write_db(
        self, instances: list[SimBotInstructionInstance], output_db_path: Path, dataset_split: str
    ) -> None:
        """Write the new dbs."""
        with DatasetDb(output_db_path, readonly=False) as write_db:
            for idx, instance in enumerate(instances):
                dataset_idx = f"{self.dataset_name}_{dataset_split}_{idx}"
                write_db[(idx, dataset_idx)] = instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_input_db_path",
        type=Path,
    )
    parser.add_argument("--train_output_db_path", type=Path)
    parser.add_argument(
        "--valid_input_db_path",
        type=Path,
    )
    parser.add_argument("--valid_output_db_path", type=Path)
    parser.add_argument(
        "--simbot_db_type", choices=["action", "cr"], help="The type of SimBot task."
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold used to match a bounding box to the target object.",
    )
    parser.add_argument(
        "--min_iou_threshold",
        type=float,
        default=0.5,
        help="Minimum IoU threshold used when match_based_on_label is True",
    )
    parser.add_argument("--match_based_on_label", action="store_true")
    args = parser.parse_args()
    db_filter = FilterSimBotDB(
        train_input_db_file=args.train_input_db_path,
        train_output_db_file=args.train_output_db_path,
        valid_input_db_file=args.valid_input_db_path,
        valid_output_db_file=args.valid_output_db_path,
        simbot_db_type=args.simbot_db_type,
        iou_threshold=args.iou_threshold,
        minimum_iou_threshold=args.min_iou_threshold,
        matching_strategy="threshold_and_label" if args.match_based_on_label else "threshold_only",
    )
    db_filter.run()
