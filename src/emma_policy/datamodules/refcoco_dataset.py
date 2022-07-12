from pathlib import Path

import torch
from emma_datasets.datamodels.datasets import RefCocoInstance
from overrides import overrides
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.utils import get_logger
from emma_policy.utils.boxes import BoxMode


logger = get_logger(__name__)


class RefCocoDataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for RefCOCOg.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_frames: int = 0,
        bbox_match_threshold: float = 0.5,
    ) -> None:

        super().__init__(
            dataset_db_path=dataset_db_path,
            tokenizer=tokenizer,
            max_frames=max_frames,
            bbox_match_threshold=bbox_match_threshold,
        )

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get the RefCOCOg instance at the given index as an instance of `EmmaDatasetItem`."""
        with self.db:
            instance_str = self.db[index]
            instance = RefCocoInstance.parse_raw(instance_str)
        return self.visual_grounding(instance)

    def visual_grounding(self, instance: RefCocoInstance) -> EmmaDatasetItem:
        """Process the instance for the RefCOCOg task."""
        source_text = self._get_random_template_for_task(Task.visual_grounding).format(
            caption=instance.referring_expression.sentence
        )

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )

        gt_box = self._get_ground_truth_bbox(instance)
        matched_index, gt_flags = self._best_match_features(
            ground_truth_bbox=gt_box.unsqueeze(0),
            object_coordinates_bbox=visual_features.object_coordinates,
            threshold=self.bbox_match_threshold,
        )

        if gt_flags[0]:
            mapped_region_index = matched_index[0]
            target_input_ids = visual_features.visual_token_ids.squeeze(0)[
                mapped_region_index
            ].reshape((1, -1))

            target_text = " ".join([self.tokenizer.decode(i) for i in target_input_ids])
            target_encoding = self.tokenizer.encode_plus(
                target_text, return_tensors=self._return_tensor_type, truncation=True
            )
            target_token_ids = target_encoding.input_ids.squeeze(0)
            decoder_attention_mask = target_encoding.attention_mask.squeeze(0)
        else:
            target_token_ids = torch.tensor(
                [
                    self.tokenizer.bos_token_id,
                    self.tokenizer.unk_token_id,
                    self.tokenizer.eos_token_id,
                ]
            )
            decoder_attention_mask = torch.ones_like(target_token_ids)

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
            task=self._get_task_as_tensor(Task.visual_grounding),
        )

    def _get_ground_truth_bbox(self, instance: RefCocoInstance) -> torch.Tensor:
        """Prepare the ground truth bounding box."""
        gt_bbox = [instance.region.x, instance.region.y, instance.region.w, instance.region.h]
        gt_bbox_coord = BoxMode.convert(
            gt_bbox, from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYXY_ABS
        )

        gt_bbox = [
            gt_bbox_coord[0] / instance.image_metadata.width,
            gt_bbox_coord[1] / instance.image_metadata.height,
            gt_bbox_coord[2] / instance.image_metadata.width,
            gt_bbox_coord[3] / instance.image_metadata.height,
        ]
        return torch.tensor(gt_bbox).clamp_(min=0.0, max=1.0)  # noqa: WPS358
