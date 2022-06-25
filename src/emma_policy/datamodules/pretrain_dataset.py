import random
from pathlib import Path
from re import sub
from typing import Any, Callable, Literal, Optional

import torch
from emma_datasets.datamodels import ActionTrajectory, DatasetMetadata, MediaType, Region
from emma_datasets.datamodels.datasets import AlfredLowAction
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetItem,
    EmmaDatasetPadding,
    EmmaVisualFeatures,
)
from emma_policy.datamodules.pretrain_instances import PretrainInstance, Task
from emma_policy.datamodules.relation import Relation
from emma_policy.utils import get_logger
from emma_policy.utils.boxes import BoxMode


log = get_logger(__name__)


def apply_token_masking(input_text: str, mlm_probability: float = 0.3) -> tuple[str, str]:
    """Applies token masking considering whole words instead of wordpieces."""
    tokens = input_text.split()
    if len(input_text) < 2 or not len(tokens):
        return "", ""

    masked_indices = torch.bernoulli(torch.full((len(tokens),), mlm_probability)).long()
    if masked_indices.sum() == 0:
        # Ensure at least one token is masked
        masked_indices[torch.randint(low=0, high=len(tokens), size=(1,))] = 1

    for idx, is_masked in enumerate(masked_indices.tolist()):
        if is_masked:
            tokens[idx] = "<mask>"

    return " ".join(tokens), input_text


def split_action_name(identifier: str) -> list[str]:
    """Split a action to lower case words."""
    # Split camel case
    matches = sub(
        "([A-Z][a-z]+)",
        r" \1",
        sub("([A-Z]+)", r" \1", identifier),
    )
    # Split "Pickup"
    matches = sub(r"(.+?)(up)($|\s)", r"\1 \2\3", matches)
    return [match.lower() for match in matches.split() if match != "Object"]


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
        max_frames: int = 0,
        bbox_match_threshold: float = 0.5,
        shuffle_frames_perc: float = 0.4,
    ) -> None:
        super().__init__(
            dataset_db_path=dataset_db_path,
            tokenizer=tokenizer,
            max_frames=max_frames,
            bbox_match_threshold=bbox_match_threshold,
            shuffle_frames_perc=shuffle_frames_perc,
        )

        self.mlm_probability = mlm_probability

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
            Task.goal_prediction: self.goal_prediction,
            Task.action_execution: self.action_execution,
            Task.vtm: self.vtm,
            Task.fom: self.fom,
            Task.vmlm: self.mlm,
        }
        self._target_padding_values = EmmaDatasetPadding().target_token_ids

    def __getitem__(self, index: int) -> Optional[EmmaDatasetItem]:
        """Get a single instance from the dataset."""
        with self.db:
            instance_str = self.db[index]
            instance = PretrainInstance.parse_raw(instance_str)
        return self.task_process_map[instance.task](instance)

    def mlm(self, instance: PretrainInstance) -> Optional[EmmaDatasetItem]:
        """Process the instance for the MLM task."""
        # applies the token masking on the original caption text
        if instance.caption is not None:
            input_text = instance.caption.text
        else:
            raise AssertionError(
                "Captions for this instance must exist. Make sure this instance is connected to the right task!"
            )

        if instance.task == Task.vmlm:
            input_text = self._refine_instruction_text(input_text)

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )

        if instance.trajectory is not None:
            action_trajectory = self._convert_trajectory_to_text(
                trajectory=instance.trajectory,
                feature_dicts=self._load_feature_dicts(instance.features_path, instance.modality),
                visual_features=visual_features,
            )
            input_text = "{input_text} {sep_token} {action_trajectory}".format(
                input_text=input_text,
                sep_token=self.tokenizer.sep_token,
                action_trajectory=action_trajectory,
            )

        source_text, target_text = apply_token_masking(input_text, self.mlm_probability)
        if not source_text:
            return None
        # formats the masked caption using the corresponding task template
        source_text = self._get_random_template_for_task(Task.mlm).format(caption=source_text)

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
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
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
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

        if other_instance.modality == MediaType.video:
            return None

        other_image_names = set(other_instance.dataset.values())
        if not image_names.isdisjoint(other_image_names):
            return None

        if other_instance.caption is None:
            return None

        return other_instance.caption.text

    def itm(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the ITM task."""
        if instance.caption is None:
            raise AssertionError(
                "Captions for this instance must exist. Make sure this instance is connected to the right task!"
            )

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

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )
        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.itm),
        )

    def visual_grounding(self, instance: PretrainInstance) -> Optional[EmmaDatasetItem]:
        """Process the instance for the Visual Grounding task."""
        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )
        feature_dict = torch.load(instance.features_path)
        if instance.regions is None:
            raise AssertionError(
                "Regions for this instance must exist. Make sure this instance is connected to the right task!"
            )
        width, height = feature_dict["width"], feature_dict["height"]
        matched_index, gt_flags = self._region_mapping(
            regions=instance.regions, visual_features=visual_features, width=width, height=height
        )
        gt_filtered = [
            reg
            for idx, reg in enumerate(instance.regions)
            if gt_flags[idx] and len(reg.caption) > 1
        ]
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

        target_text = " ".join([self.tokenizer.decode(i) for i in target_input_ids])
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            scene_features=visual_features.scene_features,
            scene_coordinates=visual_features.scene_coordinates,
            object_features=visual_features.object_features,
            object_coordinates=visual_features.object_coordinates,
            visual_token_ids=visual_features.visual_token_ids,
            scene_attention_mask=visual_features.scene_attention_mask,
            object_attention_mask=visual_features.object_attention_mask,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            object_frame_tokens=visual_features.object_frame_tokens,
            task=self._get_task_as_tensor(Task.visual_grounding),
        )

    def dense_captioning(self, instance: PretrainInstance) -> Optional[EmmaDatasetItem]:
        """Process the instance for the dense captioning task."""
        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )
        feature_dict = torch.load(instance.features_path)
        if instance.regions is None:
            raise AssertionError(
                "Regions for this instance must exist. Make sure this instance is connected to the right task!"
            )
        width, height = feature_dict["width"], feature_dict["height"]
        matched_index, gt_flags = self._region_mapping(
            regions=instance.regions, visual_features=visual_features, width=width, height=height
        )
        gt_filtered = [
            reg
            for idx, reg in enumerate(instance.regions)
            if gt_flags[idx] and len(reg.caption) > 1
        ]
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
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_text = selected_region.caption
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
            scene_frame_tokens=visual_features.scene_frame_tokens,
            object_frame_tokens=visual_features.object_frame_tokens,
            task=self._get_task_as_tensor(Task.dense_captioning),
        )

    def captioning(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the captioning task."""
        if instance.caption is None:
            raise AssertionError(
                "Captions for this instance must exist. Make sure this instance is connected to the right task!"
            )

        source_text = self._get_random_template_for_task(Task.captioning)
        target_text = instance.caption.text
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.captioning),
        )

    def vqa(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the VQA task."""
        if instance.qa_pair is None:
            raise AssertionError(
                "QA pair for this instance must exist. Make sure this instance is connected to the right task!"
            )

        input_text = instance.qa_pair.question
        target_text = instance.qa_pair.answer

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

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )

        decoder_attention_mask = target_encoding.attention_mask

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=decoder_attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.vqa),
        )

    def relation_detection(self, instance: PretrainInstance) -> Optional[EmmaDatasetItem]:
        """Process the instance for the relation detection task."""
        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )
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
            scene_frame_tokens=visual_features.scene_frame_tokens,
            object_frame_tokens=visual_features.object_frame_tokens,
            task=torch.tensor([Task.get_index(Task.relation_detection)]),
        )

    def instruction_prediction(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the instruction prediction task."""
        if instance.caption is None:
            raise AssertionError(
                "Captions for this instance must exist. Make sure this instance is connected to the right task!"
            )

        source_text = self._get_random_template_for_task(Task.instruction_prediction)
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )

        target_text = self._refine_instruction_text(instance.caption.text)

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
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.instruction_prediction),
        )

    def goal_prediction(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the instruction prediction task."""
        if instance.task_description is None:
            raise AssertionError(
                "Task description for this instance must exist. Make sure this instance is connected to the right task!"
            )

        source_text = self._get_random_template_for_task(Task.goal_prediction)
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )

        target_encoding = self.tokenizer.encode_plus(
            instance.task_description.text,
            return_tensors=self._return_tensor_type,
            truncation=True,
        )

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.instruction_prediction),
        )

    def action_execution(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the action execution task."""
        if instance.caption is None:
            raise AssertionError(
                "Captions for this instance must exist. Make sure this instance is connected to the right task!"
            )

        source_text = self._get_random_template_for_task(Task.action_execution).format(
            instruction=self._refine_instruction_text(instance.caption.text),
        )
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )

        target_text = self._convert_trajectory_to_text(
            trajectory=instance.trajectory,
            feature_dicts=self._load_feature_dicts(instance.features_path, instance.modality),
            visual_features=visual_features,
        )

        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=False
        )

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_encoding.input_ids.squeeze(0),
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.action_execution),
        )

    def vtm_negative_candidate(self, index: int, dataset_id: str) -> Optional[str]:
        """Check if the candidate is valid and return the input text.

        Args:
            index (int): Index for candidate negative sample.
            dataset_id (str): Dtaset id of positive sample.

        Returns:
            None if invalid candidate, else input text
        """
        with self.db:
            instance_str = self.db[index]
            other_instance = PretrainInstance.parse_raw(instance_str)

        if other_instance.modality == 3:
            return None

        other_dname = list(other_instance.dataset.keys())[0]
        other_dataset_id = other_instance.dataset[other_dname].id
        if dataset_id == other_dataset_id:
            return None

        if other_instance.caption is not None:
            input_text_candidates = self._refine_instruction_text(other_instance.caption.text)

            if other_instance.trajectory is not None:
                other_visual_features = self._load_visual_features(
                    features_path=other_instance.features_path, modality=other_instance.modality
                )
                other_action_trajectory = self._convert_trajectory_to_text(
                    trajectory=other_instance.trajectory,
                    feature_dicts=self._load_feature_dicts(
                        other_instance.features_path, other_instance.modality
                    ),
                    visual_features=other_visual_features,
                )

                input_text_candidates = "{input_text} {sep_token} {action_trajectory}".format(
                    input_text=input_text_candidates,
                    sep_token=self.tokenizer.sep_token,
                    action_trajectory=other_action_trajectory,
                )
        else:
            input_text_candidates = None

        return input_text_candidates

    def vtm(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the VTM task."""
        if instance.caption is None:
            raise AssertionError(
                "Captions for this instance must exist. Make sure this instance is connected to the right task!"
            )

        input_text = self._refine_instruction_text(instance.caption.text)
        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )
        target_text = "true"
        if random.random() < 0.5:  # noqa: WPS459
            target_text = "false"
            dname = list(instance.dataset.keys())[0]
            dataset_id = instance.dataset[dname].id
            rand_idx = int(len(self.db) * random.random())
            input_text = self.vtm_negative_candidate(rand_idx, dataset_id)
            while input_text is None:
                rand_idx = int(len(self.db) * random.random())
                input_text = self.vtm_negative_candidate(rand_idx, dataset_id)
        elif instance.trajectory is not None:
            action_trajectory = self._convert_trajectory_to_text(
                trajectory=instance.trajectory,
                feature_dicts=self._load_feature_dicts(instance.features_path, instance.modality),
                visual_features=visual_features,
            )
            input_text = "{input_text} {sep_token} {action_trajectory}".format(
                input_text=input_text,
                sep_token=self.tokenizer.sep_token,
                action_trajectory=action_trajectory,
            )

        source_text = self._get_random_template_for_task(task=Task.vtm).format(
            statement=input_text,
        )

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )

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
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.vtm),
        )

    def fom(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the FOM task."""
        if instance.caption is None:
            raise AssertionError(
                "Captions for this instance must exist. Make sure this instance is connected to the right task!"
            )

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality, shuffle_frames=True
        )
        original_order = visual_features.original_frame_order
        input_text = self._refine_instruction_text(instance.caption.text)

        if instance.trajectory is not None:
            ordered_features = self._load_visual_features(
                features_path=instance.features_path, modality=instance.modality
            )
            action_trajectory = self._convert_trajectory_to_text(
                trajectory=instance.trajectory,
                feature_dicts=self._load_feature_dicts(instance.features_path, instance.modality),
                visual_features=ordered_features,
            )
            input_text = "{input_text} {sep_token} {action_trajectory}".format(
                input_text=input_text,
                sep_token=self.tokenizer.sep_token,
                action_trajectory=action_trajectory,
            )
        source_text = self._get_random_template_for_task(Task.fom).format(instruction=input_text)
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        target_token_ids, decoder_attention_mask = self._create_masked_fom_targets(
            original_order=original_order,
            frame_tokens=visual_features.scene_frame_tokens.squeeze(0),
        )
        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_token_ids,
            decoder_attention_mask=decoder_attention_mask,
            scene_features=visual_features.scene_features,
            scene_coordinates=visual_features.scene_coordinates,
            object_features=visual_features.object_features,
            object_coordinates=visual_features.object_coordinates,
            visual_token_ids=visual_features.visual_token_ids,
            scene_attention_mask=visual_features.scene_attention_mask,
            object_attention_mask=visual_features.object_attention_mask,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            object_frame_tokens=visual_features.object_frame_tokens,
            task=self._get_task_as_tensor(Task.fom),
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
            threshold=self.bbox_match_threshold,
        )
        return matched_index, gt_flags

    def _get_object_class_from_action(self, low_action: AlfredLowAction) -> str:
        """Returns the class of the object that the agent is manipulating.

        We assume that a manipulation action will have a field `object_id` that indicates the
        object we're manipulating. The behaviour of the action `PutObject` is a bit
        counterintuitive because we don't have to specify the receptacle (i.e., where you want to
        place the object that the robot is holding). See the AI2Thor documentation for more
        details: https://ai2thor.allenai.org/ithor/documentation/interactive-physics/#sub-put-object
        """
        if low_action.api_action.action == "PutObject":
            return low_action.api_action.receptacle_object_id.split("|", 1)[0]
        return low_action.api_action.object_id.split("|", 1)[0]

    def _convert_trajectory_to_text(
        self,
        trajectory: ActionTrajectory,
        feature_dicts: list[dict[str, Any]],
        visual_features: EmmaVisualFeatures,
        truncation_side: Literal["left", "right"] = "left",
    ) -> str:
        """Convert an Alfred trajectory to text.

        If an object is not found, the `<unk>` token is used.
        """
        low_level_actions = trajectory.low_level_actions

        if self.max_frames:
            feature_dicts = self._truncate_frames(feature_dicts, truncation_side=truncation_side)
            low_level_actions = self._truncate_frames(
                low_level_actions, truncation_side=truncation_side
            )

        trajectory_text = []
        for action_idx, action in enumerate(low_level_actions):
            # Split a cama case action to words
            trajectory_text.extend(split_action_name(action.api_action.action))
            # Match the object to a predicted bounding box
            if "bbox" in action.discrete_action.args:

                bbox_coord = action.discrete_action.args["bbox"]  # noqa: WPS529
                gt_bbox = torch.tensor(
                    [
                        bbox_coord[0] / feature_dicts[action_idx]["width"],
                        bbox_coord[1] / feature_dicts[action_idx]["height"],
                        bbox_coord[2] / feature_dicts[action_idx]["width"],
                        bbox_coord[3] / feature_dicts[action_idx]["height"],
                    ]
                )

                # Get the index of the objects from the current frame. Frames start from 1.
                frame_token = self.tokenizer.convert_tokens_to_ids(f"<frame_token_{action_idx+1}>")
                frame_objects = visual_features.object_frame_tokens == frame_token

                matched_index, gt_flags = self._best_match_features(
                    ground_truth_bbox=gt_bbox.unsqueeze(0),
                    object_coordinates_bbox=visual_features.object_coordinates[frame_objects],
                    threshold=self.bbox_match_threshold,
                )

                # we first add the class of the object we want to interact with
                # reference object is always the first argument of a discrete action
                object_class = self._get_object_class_from_action(action)
                trajectory_text.append(object_class.lower())

                # then if we have a matching bounding box, we add the visual token as well
                found_matched_object = gt_flags[0]
                if found_matched_object:
                    trajectory_text.append(
                        self.tokenizer.decode(
                            visual_features.visual_token_ids[frame_objects][matched_index[0]]
                        )
                    )

            trajectory_text.append(self.tokenizer.sep_token)

        return " ".join(trajectory_text)

    def _create_masked_fom_targets(
        self,
        original_order: torch.Tensor,
        frame_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create masked fom targets.

        Mask tokens where the original frame order matches the shuffled one.
        """
        sequence_order = torch.argsort(original_order)
        order_encoded = frame_tokens[sequence_order]
        target_text = "".join([self.tokenizer.decode(i) for i in order_encoded])
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_token_ids = target_encoding.input_ids.squeeze(0)
        decoder_attention_mask = target_encoding.attention_mask.squeeze(0)
        # Add +1 because tokenizer.encode_plus has added the sos token
        mask_inidices = torch.where(original_order == sequence_order)[0] + 1
        target_token_ids[mask_inidices] = self._target_padding_values
        return target_token_ids, decoder_attention_mask
