import argparse
from typing import Any, Union

import torch
from emma_datasets.constants.simbot.simbot import get_arena_definitions
from emma_datasets.datamodels.datasets.simbot import SimBotInstructionInstance
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_asset_from_object_id,
)
from emma_datasets.db import DatasetDb
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn import CosineSimilarity
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

from emma_policy.datamodules.base_dataset import best_match_features
from emma_policy.utils import decompress_simbot_mask


class EntityFeatureClassifier:
    """EntityFeatureClassifier class.

    Used to determine the sub-classes based on pure feature averaging and cosine similarity.
    """

    def __init__(
        self,
        train_db: str,
        test_db: str,
        save_path: str,
        feature_size: float = 2048,
    ) -> None:
        self.train_db = train_db
        self.test_db = test_db
        self.save_path = save_path

        arena_definitions = get_arena_definitions()
        self._assets_to_labels = get_arena_definitions()["asset_to_label"]
        self._special_asset_to_readable_name = arena_definitions["special_asset_to_readable_name"]
        self._remaining_subclasses = {
            "Computer_Monitor_01": "Computer",
            "Computer_Monitor_Broken": "Computer",
            "Computer_Monitor_New": "Computer",
            "Lab_Terminal": "Computer",
            "TAMPrototypeHead_01": "Computer",
            "Desk_01": "Table",
            "SM_Prop_Table_02": "Table",
        }
        self._special_asset_to_readable_name.update(self._remaining_subclasses)
        self._special_asset_running_avg = {
            readable_name: {"count": 1, "centroid": torch.zeros(feature_size)}  # type: ignore[call-overload]
            for _, readable_name in self._special_asset_to_readable_name.items()
        }
        self._similarity = CosineSimilarity()

    def run(self) -> None:
        """Run the classification pipeline."""
        self.run_for_split(input_db=self.train_db, split="train")
        self._save_centroids()
        if self.test_db is not None:
            self._test_results: dict[str, list[str]] = {"groundtruth": [], "prediction": []}
            self.run_for_split(input_db=self.test_db, split="test")
            report = classification_report(
                self._test_results["groundtruth"], self._test_results["prediction"]
            )
            print(report)  # noqa: WPS421
            cm = confusion_matrix(
                self._test_results["groundtruth"], self._test_results["prediction"]
            )
            print(cm)  # noqa: WPS421

    def run_for_split(self, input_db: str, split: str = "train") -> None:
        """Run the classifier for a single split."""
        db = DatasetDb(input_db)
        db_size = len(db)

        for idx in tqdm(range(db_size)):
            instance_str = db[idx]
            instance = SimBotInstructionInstance.parse_raw(instance_str)
            for action_idx, action in enumerate(instance.actions):
                action_type = action.type
                action_metadata = getattr(action, action_type.lower())
                action_object_metadata = action_metadata.get("object", None)

                # Ignore actions that do not have an object id
                if (  # noqa: WPS337
                    action_type == "Examine"
                    or action_object_metadata is None
                    or "id" not in action_object_metadata
                    or action_object_metadata.get("mask", None) is None
                ):
                    continue

                features = torch.load(instance.features_path[action_idx])["frames"][0]["features"]
                self._run_for_action(
                    action_type,
                    action_object_metadata,
                    features,
                    instance.vision_augmentation,
                    split,
                )

    def _run_for_object(
        self,
        object_asset: str,
        object_mask: Union[list[int], list[list[int]]],
        features: dict[str, Any],
        vision_augmentation: bool = False,
        split: str = "train",
    ) -> None:
        if object_asset not in self._special_asset_to_readable_name:
            return

        readable_name = self._special_asset_to_readable_name[object_asset]
        matched_indices, ground_truth_flags = self._gt_bbox_from_features(
            object_mask, features, vision_augmentation
        )

        if split == "train" and ground_truth_flags[0].item():
            self._update_running_average(
                object_asset=readable_name,
                bbox_features=features["bbox_features"][matched_indices[0].item()],
            )

        elif split == "test" and ground_truth_flags[0].item():
            self._update_test_metrics(
                object_asset=readable_name,
                bbox_features=features["bbox_features"][matched_indices[0].item()],
            )

    def _run_for_action(
        self,
        action_type: str,
        action_object_metadata: dict[str, Any],
        features: dict[str, Any],
        vision_augmentation: bool = False,
        split: str = "train",
    ) -> None:
        if action_type == "Search":
            for object_idx, object_id in enumerate(action_object_metadata["id"]):
                object_asset = get_object_asset_from_object_id(object_id, self._assets_to_labels)
                self._run_for_object(
                    object_asset=object_asset,
                    object_mask=action_object_metadata["mask"][object_idx],
                    features=features,
                    vision_augmentation=vision_augmentation,
                )

        else:
            object_asset = get_object_asset_from_object_id(
                action_object_metadata["id"], self._assets_to_labels
            )
            self._run_for_object(
                object_asset=object_asset,
                object_mask=action_object_metadata["mask"],
                features=features,
                vision_augmentation=vision_augmentation,
                split=split,
            )

    def _gt_bbox_from_features(
        self,
        mask: Union[list[int], list[list[int]]],
        features: dict[str, Any],
        vision_augmentation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vision_augmentation:
            gt_bbox = torch.tensor(mask).unsqueeze(0)
        else:
            gt_binary_mask = decompress_simbot_mask(mask)  # type: ignore[arg-type]
            gt_bbox = masks_to_boxes(torch.tensor(gt_binary_mask).unsqueeze(0))

        return best_match_features(
            ground_truth_bbox=gt_bbox,
            object_coordinates_bbox=features["bbox_coords"],
            threshold=0.5,
        )

    def _update_running_average(self, object_asset: str, bbox_features: torch.Tensor) -> None:
        running_counter = self._special_asset_running_avg[object_asset]["count"]
        self._special_asset_running_avg[object_asset]["centroid"] = (
            (running_counter - 1) * self._special_asset_running_avg[object_asset]["centroid"]
            + bbox_features
        ) / running_counter
        self._special_asset_running_avg[object_asset]["count"] += 1

    def _update_test_metrics(self, object_asset: str, bbox_features: torch.Tensor) -> None:
        assets = list(self._special_asset_running_avg.keys())
        centroids = [
            special_asset_dict["centroid"]
            for special_asset_dict in list(self._special_asset_running_avg.values())
        ]
        similarity_value = self._similarity(bbox_features, torch.stack(centroids))
        most_similar_vector = similarity_value.argmax().item()

        predicted_asset = assets[most_similar_vector]
        self._test_results["groundtruth"].append(object_asset)
        self._test_results["prediction"].append(predicted_asset)

    def _save_centroids(self) -> None:
        centroids = {}
        for special_asset, special_asset_dict in self._special_asset_running_avg.items():
            # Ignore assets that we didnt find any examples during train
            if special_asset_dict["count"] > 1:
                centroids[special_asset] = special_asset_dict["centroid"]
        readable_to_class = {
            readable_name: self._assets_to_labels.get(asset, asset)
            for asset, readable_name in self._special_asset_to_readable_name.items()
        }

        centroids_per_class: dict[str, dict[str, torch.Tensor]] = {}
        for readable_name, centroid in centroids.items():
            object_class = readable_to_class[readable_name]
            if object_class not in centroids_per_class:
                centroids_per_class[object_class] = {}
            centroids_per_class[object_class][readable_name] = centroid

        torch.save(centroids_per_class, self.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_db",
        help="Path to train db.",
    )
    parser.add_argument(
        "--test_db",
        help="Path to test db.",
    )
    parser.add_argument(
        "--save_path",
        help="Path to output image.",
    )
    args = parser.parse_args()

    classifier = EntityFeatureClassifier(
        train_db=args.train_db,
        test_db=args.test_db,
        save_path=args.save_path,
    )

    classifier.run()
