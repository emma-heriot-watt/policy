from pathlib import Path

import torch
from emma_datasets.datamodels.datasets import VQAv2Instance
from overrides import overrides
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.utils import get_logger


logger = get_logger(__name__)


class VQAv2Dataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for VQA-v2.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_frames: int = 0,
    ) -> None:
        super().__init__(
            dataset_db_path=dataset_db_path,
            tokenizer=tokenizer,
            max_frames=max_frames,
        )

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get the VQA-v2 instance at the given index as an instance of `EmmaDatasetItem`."""
        with self.db:
            instance_str = self.db[index]
            instance = VQAv2Instance.parse_raw(instance_str)
        return self.vqa(instance)

    def vqa(self, instance: VQAv2Instance) -> EmmaDatasetItem:
        """Process the instance for the VQA-v2 task."""
        source_text = self._get_random_template_for_task(Task.vqa).format(
            question=instance.question
        )
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )

        if instance.training_targets:
            # Select the target answer based on the answer scores
            scores = torch.tensor([answer.score for answer in instance.training_targets])
            selected_idx = torch.multinomial(scores, 1).item()
            target_text = instance.training_targets[selected_idx].answer
        else:
            target_text = ""

        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target = {"question_id": instance.question_id, "answers": instance.answers}
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
            task=self._get_task_as_tensor(Task.vqa),
            raw_target=target,
        )
