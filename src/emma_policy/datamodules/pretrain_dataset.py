import random
from pathlib import Path
from typing import Callable

import torch
from emma_datasets.db import DatasetDb
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem, EmmaVisualFeatures
from emma_policy.datamodules.pretrain_instances import PretrainInstance, Task
from emma_policy.datamodules.pretrain_instances.datamodels import TASK_TEMPLATES_MAP
from emma_policy.utils import get_logger


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


class EmmaPretrainDataset(Dataset[EmmaDatasetItem]):
    """Pretrain dataset reader for the EMMA model.

    Each task in the `self.task_process_map` corresponds to a method which will take the instance
    and return an instance of the `EmmaDatasetItem`.
    """

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        mlm_probability: float = 0.3,
    ) -> None:
        self.db = DatasetDb(dataset_db_path)
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

        self.task_process_map: dict[Task, Callable[[PretrainInstance], EmmaDatasetItem]] = {
            Task.mlm: self.mlm,
            Task.itm: self.itm,
            Task.visual_grounding: self.visual_grounding,
            Task.dense_captioning: self.dense_captioning,
            Task.captioning: self.captioning,
            Task.vqa: self.vqa,
            Task.instruction_prediction: self.instruction_prediction,
            Task.action_execution: self.action_execution,
            Task.vtm: self.vtm,
        }

    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return len(self.db)

    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get a single instance from the dataset."""
        with self.db:
            instance_str = self.db[index]
            instance = PretrainInstance.parse_raw(instance_str)

        return self.task_process_map[instance.task](instance)

    def load_visual_features(self, instance: PretrainInstance) -> EmmaVisualFeatures:
        """Generate all the required visual features for the current instance."""
        features_path = instance.features_path

        if instance.modality == 4:
            feature_dicts = [
                feature_dict["features"] for feature_dict in torch.load(features_path)["frames"]
            ]
        elif instance.modality == 3:
            feature_dicts = [torch.load(features_path)]

        object_features = []
        object_coordinates = []
        scene_features = []
        vis_tokens = []
        obj_frame_ids = []
        object_attention_mask = []

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
                )
            )

            vis_tokens.append(curr_vis_tokens)

            obj_frame_ids.append(
                curr_vis_tokens.new_full(curr_vis_tokens.shape, fill_value=frame_idx + 1)
            )

            object_attention_mask.append(torch.ones_like(curr_vis_tokens))

        num_frames = len(scene_features)
        scene_attention_mask = torch.ones(num_frames)
        scene_coordinates = torch.tensor([0, 0, 1.0, 1.0]).repeat(num_frames, 1)
        scene_frame_ids = torch.arange(1, num_frames + 1)

        emma_visual_features = EmmaVisualFeatures(
            object_attention_mask=torch.cat(object_attention_mask),
            object_coordinates=torch.cat(object_coordinates),
            object_features=torch.cat(object_features),
            object_frame_ids=torch.cat(obj_frame_ids),
            scene_attention_mask=scene_attention_mask,
            scene_coordinates=scene_coordinates,
            scene_features=torch.cat(scene_features),
            scene_frame_ids=scene_frame_ids,
            visual_token_ids=torch.cat(vis_tokens),
        )

        return emma_visual_features

    def mlm(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the MLM task."""
        # applies the token masking on the original caption text
        if instance.caption is not None:
            input_text = instance.caption.text
        else:
            input_text = "the monkey is eating a banana."

        source_text, target_text = apply_token_masking(input_text, self.mlm_probability)
        # formats the masked caption using the corresponding task template
        source_text = random.choice(TASK_TEMPLATES_MAP[Task.mlm]).format(  # noqa: S311
            caption=source_text
        )

        input_encoding = self.tokenizer.encode_plus(source_text, return_tensors="pt")
        target_encoding = self.tokenizer.encode_plus(target_text, return_tensors="pt")

        visual_features = self.load_visual_features(instance)

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
        )

    def itm(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the ITM task."""
        # raise NotImplementedError
        return self.mlm(instance)

    def visual_grounding(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the Visual Grounding task."""
        # raise NotImplementedError
        return self.mlm(instance)

    def dense_captioning(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the dense captioning task."""
        # raise NotImplementedError
        return self.mlm(instance)

    def captioning(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the captioning task."""
        return self.mlm(instance)

        # source_text = random.choice(TASK_TEMPLATES_MAP[Task.captioning])
        # target_text = instance.caption.text

        # input_encoding = self.tokenizer.encode_plus(source_text)
        # target_encoding = self.tokenizer.encode_plus(target_text)

        # visual_features = self.load_visual_features(instance=instance)

        # return EmmaDatasetItem(
        #     input_token_ids=input_encoding.input_ids,
        #     target_token_ids=target_encoding.input_ids,
        #     scene_features=visual_features.scene_features,
        #     scene_coordinates=visual_features.scene_coordinates,
        #     object_features=visual_features.object_features,
        #     object_coordinates=visual_features.object_coordinates,
        #     visual_token_ids=visual_features.visual_token_ids,
        # )

    def vqa(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the VQA task."""
        # raise NotImplementedError
        return self.mlm(instance)

    def instruction_prediction(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the instruction prediction task."""
        # raise NotImplementedError
        return self.mlm(instance)

    def action_execution(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the action execution task."""
        # raise NotImplementedError
        return self.mlm(instance)

    def vtm(self, instance: PretrainInstance) -> EmmaDatasetItem:
        """Process the instance for the VTM task."""
        # raise NotImplementedError
        return self.mlm(instance)
