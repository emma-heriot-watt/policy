import random
from pathlib import Path
from re import finditer
from typing import Callable, Optional

import torch
from emma_datasets.datamodels import DatasetMetadata, Region
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem, EmmaVisualFeatures
from emma_policy.datamodules.pretrain_instances import PretrainInstance, Task
from emma_policy.datamodules.relation import Relation
from emma_policy.utils import get_logger
from emma_policy.utils.boxes import BoxMode


log = get_logger(__name__)


def apply_token_masking(input_text: str, mlm_probability: float = 0.3) -> tuple[str, str]:
    """Applies token masking considering whole words instead of wordpieces."""
    tokens = input_text.split()

    masked_indices = torch.bernoulli(torch.full((len(tokens),), mlm_probability)).long()

    if masked_indices.sum() == 0:
        return input_text, input_text

    for idx, is_masked in enumerate(masked_indices.tolist()):
        if is_masked:
            tokens[idx] = "<mask>"

    return " ".join(tokens), input_text


def camel_case_split(identifier: str) -> list[str]:
    """Split a camel case action to lower case words."""
    matches = finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier)
    return [match.group(0).lower() for match in matches if match.group(0) != "Object"]


class EmmaPretrainDataset(EmmaBaseDataset[Optional[EmmaDatasetItem]]):
    """Pretrain dataset reader for the EMMA model.

    Each task in the `self.task_process_map` corresponds to a method which will take the instance
    and return an instance of the `EmmaDatasetItem`.
    """

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        mlm_probability: float = 0.3,
        max_frames: Optional[int] = None,
        match_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            dataset_db_path=dataset_db_path,
            tokenizer=tokenizer,
            max_frames=max_frames,
        )

        self.mlm_probability = mlm_probability
        self.match_threshold = match_threshold

        self.task_process_map: dict[
            Task, Callable[[PretrainInstance], Optional[EmmaDatasetItem]]
        ] = {
            Task.mlm: self.mlm,
            Task.itm: self.itm,
            Task.visual_grounding: self.visual_grounding,
            Task.dense_captioning: self.dense_captioning,
            Task.relation_detection: self.relation_detection,
            Task.captioning: self.captioning,
            Task.vqa: self.vqa,
            Task.instruction_prediction: self.instruction_prediction,
            Task.action_execution: self.action_execution,
            Task.vtm: self.vtm,
        }

    def __getitem__(self, index: int) -> Optional[EmmaDatasetItem]:
        """Get a single instance from the dataset."""
        with self.db:
            instance_str = self.db[index]
            instance = PretrainInstance.parse_raw(instance_str)

        return self.task_process_map[instance.task](instance)

    def mlm(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the MLM task."""
        # applies the token masking on the original caption text
        if instance.caption is not None:
            input_text = instance.caption.text
        else:
            raise AssertionError(
                "Captions for this instance must exist. Make sure this instance is connected to the right task!"
            )

        source_text, target_text = apply_token_masking(input_text, self.mlm_probability)
        # formats the masked caption using the corresponding task template
        source_text = self._get_random_template_for_task(Task.mlm).format(caption=source_text)

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        visual_features = self._load_visual_features(instance)

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_ids=visual_features.object_frame_ids,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_ids=visual_features.scene_frame_ids,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.mlm),
        )

    def itm_negative_candidate(
        self, index: int, image_names: set[DatasetMetadata]
    ) -> Optional[str]:
        """Check if the candidate is valid and return the input text.

        Args:
            index (int): Index for candidate negative sample.
            image_names (set): Name of image in all datasets.

        Returns:
            None if invalid candidate, else input text
        """
        with self.db:
            instance_str = self.db[index]
            other_instance = PretrainInstance.parse_raw(instance_str)

        if other_instance.modality == 4:
            return None

        other_image_names = set(other_instance.dataset.values())
        if not image_names.isdisjoint(other_image_names):
            return None

        if other_instance.caption is not None:
            input_text_candidates = other_instance.caption.text
        elif other_instance.regions is not None:
            input_text_candidates = other_instance.regions.caption
        else:
            input_text_candidates = None

        return input_text_candidates

    def itm(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the ITM task."""
        input_text = instance.caption.text
        target_text = "true"
        if random.random() < 0.5:  # noqa: WPS459
            target_text = "false"
            img_names = set(instance.dataset.values())
            rand_idx = int(len(self.db) * random.random())
            input_text = self.itm_negative_candidate(rand_idx, img_names)
            while input_text is None:
                rand_idx = int(len(self.db) * random.random())
                input_text = self.itm_negative_candidate(rand_idx, img_names)

        # formats the masked caption using the corresponding task template
        input_text = self._get_random_template_for_task(Task.itm).format(
            statement=input_text.strip(".")
        )

        input_encoding = self.tokenizer.encode_plus(
            input_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type
        )

        visual_features = self._load_visual_features(instance)

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_ids=visual_features.object_frame_ids,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_ids=visual_features.scene_frame_ids,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.itm),
        )

    def _region_mapping(
        self,
        regions: list[Region],
        visual_features: EmmaVisualFeatures,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        gt_bbox = []
        for region in regions:
            gt_bbox_coord = BoxMode.convert(
                list(region.bbox), from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYXY_ABS
            )

            gt_bbox.append(
                [
                    gt_bbox_coord[0] / width,
                    gt_bbox_coord[1] / height,
                    gt_bbox_coord[2] / width,
                    gt_bbox_coord[3] / height,
                ]
            )

        matched_index, gt_flags = self._best_match_features(
            ground_truth_bbox=torch.tensor(gt_bbox),
            object_coordinates_bbox=visual_features.object_coordinates,
            threshold=self.match_threshold,
        )
        return matched_index, gt_flags

    def visual_grounding(self, instance: PretrainInstance) -> Optional[EmmaDatasetItem]:
        """Process the instance for the Visual Grounding task."""
        visual_features = self._load_visual_features(instance=instance)
        feature_dict = torch.load(instance.features_path)
        if instance.regions is None:
            raise AssertionError(
                "Regions for this instance must exist. Make sure this instance is connected to the right task!"
            )
        width, height = feature_dict["width"], feature_dict["height"]
        matched_index, gt_flags = self._region_mapping(
            regions=instance.regions, visual_features=visual_features, width=width, height=height
        )

        gt_filtered = [reg for idx, reg in enumerate(instance.regions) if gt_flags[idx]]
        if not gt_filtered:
            return None
        rand_index = random.randint(0, len(gt_filtered) - 1)
        selected_region = gt_filtered[rand_index]
        mapped_region_index = matched_index[gt_flags][rand_index]

        source_text = self._get_random_template_for_task(Task.visual_grounding).format(
            caption=selected_region.caption
        )

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_input_ids = visual_features.visual_token_ids.squeeze(0)[
            mapped_region_index
        ].reshape((1, -1))
        decoder_attention_mask = torch.ones(target_input_ids.shape, dtype=torch.bool)

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_input_ids.squeeze(0),
            decoder_attention_mask=decoder_attention_mask.squeeze(0),
            scene_features=visual_features.scene_features,
            scene_coordinates=visual_features.scene_coordinates,
            object_features=visual_features.object_features,
            object_coordinates=visual_features.object_coordinates,
            visual_token_ids=visual_features.visual_token_ids,
            scene_attention_mask=visual_features.scene_attention_mask,
            object_attention_mask=visual_features.object_attention_mask,
            scene_frame_ids=visual_features.scene_frame_ids,
            object_frame_ids=visual_features.object_frame_ids,
            task=self._get_task_as_tensor(Task.visual_grounding),
        )

    def dense_captioning(self, instance: PretrainInstance) -> Optional[EmmaDatasetItem]:
        """Process the instance for the dense captioning task."""
        visual_features = self._load_visual_features(instance=instance)
        feature_dict = torch.load(instance.features_path)
        if instance.regions is None:
            raise AssertionError(
                "Regions for this instance must exist. Make sure this instance is connected to the right task!"
            )
        width, height = feature_dict["width"], feature_dict["height"]
        matched_index, gt_flags = self._region_mapping(
            regions=instance.regions, visual_features=visual_features, width=width, height=height
        )
        gt_filtered = [reg for idx, reg in enumerate(instance.regions) if gt_flags[idx]]
        if not gt_filtered:
            return None
        rand_index = random.randint(0, len(gt_filtered) - 1)
        selected_region = gt_filtered[rand_index]
        mapped_region_index = matched_index[gt_flags][rand_index]

        source_caption = self.tokenizer.decode(
            visual_features.visual_token_ids.squeeze(0)[mapped_region_index]
        )

        source_text = self._get_random_template_for_task(Task.dense_captioning).format(
            region=source_caption
        )

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type
        )
        target_text = selected_region.caption
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type
        )

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            scene_features=visual_features.scene_features,
            scene_coordinates=visual_features.scene_coordinates,
            object_features=visual_features.object_features,
            object_coordinates=visual_features.object_coordinates,
            visual_token_ids=visual_features.visual_token_ids,
            scene_attention_mask=visual_features.scene_attention_mask,
            object_attention_mask=visual_features.object_attention_mask,
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            scene_frame_ids=visual_features.scene_frame_ids,
            object_frame_ids=visual_features.object_frame_ids,
            task=self._get_task_as_tensor(Task.dense_captioning),
        )

    def captioning(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the captioning task."""
        source_text = self._get_random_template_for_task(Task.captioning)
        target_text = instance.caption.text

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        visual_features = self._load_visual_features(instance)

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_ids=visual_features.object_frame_ids,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_ids=visual_features.scene_frame_ids,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.captioning),
        )

    def vqa(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the VQA task."""
        input_text = instance.qa.question
        target_text = instance.qa.answer

        # formats the masked caption using the corresponding task template
        source_text = self._get_random_template_for_task(Task.vqa).format(
            question=input_text,
        )

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        visual_features = self._load_visual_features(instance)

        decoder_attention_mask = target_encoding.attention_mask

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=decoder_attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_ids=visual_features.object_frame_ids,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_ids=visual_features.scene_frame_ids,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.vqa),
        )

    def _relation_detection_target(self, selected_relation: Relation) -> str:
        """Generate the target for relation detection task."""
        subject_attr_list = selected_relation.subject_attr
        if subject_attr_list:
            subject_attr = " ".join(
                random.sample(subject_attr_list, min(len(subject_attr_list), 2))
            )
        else:
            subject_attr = ""

        object_attr_list = selected_relation.object_attr

        if object_attr_list:
            object_attr = " ".join(random.sample(object_attr_list, min(len(object_attr_list), 2)))
        else:
            object_attr = ""

        subject_type = selected_relation.subject.caption
        object_type = selected_relation.object.caption
        predicate = selected_relation.predicate
        target_text = f"{subject_attr} {subject_type} {predicate} {object_attr} {object_type}"
        target_text = target_text.strip().replace("  ", " ")
        return target_text

    def relation_detection(self, instance: PretrainInstance) -> Optional[EmmaDatasetItem]:
        """Process the instance for the relation detection task."""
        visual_features = self._load_visual_features(instance=instance)
        feature_dict = torch.load(instance.features_path)
        if instance.relations is None:
            raise AssertionError(
                "relation for this instance must exist. Make sure this instance is connected to the right task!"
            )

        sub_matched_index, sub_gt_flags = self._region_mapping(
            regions=[rel.subject for rel in instance.relations],
            visual_features=visual_features,
            width=feature_dict["width"],
            height=feature_dict["height"],
        )
        obj_matched_index, obj_gt_flags = self._region_mapping(
            regions=[rel.object for rel in instance.relations],
            visual_features=visual_features,
            width=feature_dict["width"],
            height=feature_dict["height"],
        )

        combined_flag = torch.logical_and(sub_gt_flags, obj_gt_flags)
        gt_filtered = [rel for idx, rel in enumerate(instance.relations) if combined_flag[idx]]
        if not gt_filtered:
            return None
        rand_index = random.randint(0, len(gt_filtered) - 1)
        mapped_subj_index = sub_matched_index[combined_flag][rand_index]
        mapped_obj_index = obj_matched_index[combined_flag][rand_index]

        subject_token = self.tokenizer.decode(
            visual_features.visual_token_ids.squeeze(0)[mapped_subj_index]
        )

        object_token = self.tokenizer.decode(
            visual_features.visual_token_ids.squeeze(0)[mapped_obj_index]
        )

        source_text = self._get_random_template_for_task(Task.relation_detection).format(
            subject=subject_token, object=object_token
        )
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_text = self._relation_detection_target(selected_relation=gt_filtered[rand_index])
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            scene_features=visual_features.scene_features,
            scene_coordinates=visual_features.scene_coordinates,
            object_features=visual_features.object_features,
            object_coordinates=visual_features.object_coordinates,
            visual_token_ids=visual_features.visual_token_ids,
            scene_attention_mask=visual_features.scene_attention_mask,
            object_attention_mask=visual_features.object_attention_mask,
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            scene_frame_ids=visual_features.scene_frame_ids,
            object_frame_ids=visual_features.object_frame_ids,
            task=torch.tensor([Task.get_index(Task.relation_detection)]),
        )

    def convert_trajectory_to_text(
        self, instance: PretrainInstance, visual_features: EmmaVisualFeatures
    ) -> Optional[str]:
        """Convert an Alfred trajectory to text."""
        feature_dicts = torch.load(instance.features_path)["frames"]

        trajectory = []
        for action_idx, action in enumerate(instance.trajectory.low_level_actions):
            if action_idx > 0:
                trajectory.append(self.tokenizer.eos_token)

            # Split a cama case action to words
            trajectory.extend(camel_case_split(action.api_action.action))
            # Match the object to a predicted bounding box
            if "bbox" in action.discrete_action.args:

                bbox_coord = BoxMode.convert(
                    list(action.discrete_action.args["bbox"]),  # noqa: WPS529
                    from_mode=BoxMode.XYWH_ABS,
                    to_mode=BoxMode.XYXY_ABS,
                )
                gt_bbox = torch.tensor(
                    [
                        bbox_coord[0] / feature_dicts[action_idx]["features"]["width"],
                        bbox_coord[1] / feature_dicts[action_idx]["features"]["height"],
                        bbox_coord[2] / feature_dicts[action_idx]["features"]["width"],
                        bbox_coord[3] / feature_dicts[action_idx]["features"]["height"],
                    ]
                )

                # Get the index of the objects from the current frame. Frames start from 1.
                frame_objects = visual_features.object_frame_ids == action_idx + 1
                matched_index, gt_flags = self._best_match_features(
                    ground_truth_bbox=gt_bbox.unsqueeze(0),
                    object_coordinates_bbox=visual_features.object_coordinates[frame_objects],
                    threshold=self.match_threshold,
                )
                if not gt_flags[0]:
                    return None
                trajectory.append(
                    self.tokenizer.decode(
                        visual_features.visual_token_ids[frame_objects][matched_index[0]]
                    )
                )

        return " ".join(trajectory)

    def instruction_prediction(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the instruction prediction task."""
        raise NotImplementedError

    def action_execution(self, instance: PretrainInstance) -> Optional[EmmaDatasetItem]:
        """Process the instance for the action execution task."""
        source_text = self._get_random_template_for_task(Task.action_execution).format(
            instruction=instance.caption.text,
        )
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        visual_features = self._load_visual_features(instance)

        target_text = self.convert_trajectory_to_text(instance, visual_features)
        if target_text is None:
            return None
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_ids=visual_features.object_frame_ids,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_ids=visual_features.scene_frame_ids,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.action_execution),
        )

    def vtm(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the VTM task."""
        raise NotImplementedError
