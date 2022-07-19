from pathlib import Path
from typing import Optional

from emma_datasets.datamodels.datasets.coco import CocoInstance
from overrides import overrides
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.utils import get_logger


logger = get_logger(__name__)


class COCOCaptioningDataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for COCO captioning.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_frames: int = 0,
        merged_annotations: bool = True,
        is_train: bool = True,
    ) -> None:

        if not merged_annotations:
            raise NotImplementedError(
                "Expecting dbs where every instance is an image associated with all of its captions."
            )

        super().__init__(
            dataset_db_path=dataset_db_path, tokenizer=tokenizer, max_frames=max_frames
        )

        self.is_train = is_train
        if is_train:
            index_db_map, dataset_size = self._unpack_annotations()
            self.index_db_map = index_db_map
            self.dataset_size = dataset_size
        else:
            self.dataset_size = len(self.db)

    @overrides(check_signature=False)
    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return self.dataset_size

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get the COCO instance at the given index as an instance of `EmmaDatasetItem`."""
        instance_str: str
        if self.is_train:
            db_map = self.index_db_map[index]
            with self.db:
                instance_str = self.db[db_map["db_index"]]

            instance = CocoInstance.parse_raw(instance_str)
            return self.captioning(instance, caption_index=db_map["caption_index"])

        with self.db:
            instance_str = self.db[index]

        instance = CocoInstance.parse_raw(instance_str)
        return self.captioning(instance)

    def captioning(
        self, instance: CocoInstance, caption_index: Optional[int] = None
    ) -> EmmaDatasetItem:
        """Process the instance for the COCO captioning task."""
        if instance.captions is None:
            raise AssertionError(
                "Captions for this instance must exist. Make sure this instance is connected to the right task!"
            )

        if self.is_train:
            source_text = self._get_random_template_for_task(Task.captioning)

            input_encoding = self.tokenizer.encode_plus(
                source_text, return_tensors=self._return_tensor_type, truncation=True
            )

            target_token_ids = None
            decoder_attention_mask = None

            target_text = instance.captions[caption_index].strip()

            target_encoding = self.tokenizer.encode_plus(
                target_text, return_tensors=self._return_tensor_type, truncation=True
            )
            target_token_ids = target_encoding.input_ids.squeeze(0)

            decoder_attention_mask = target_encoding.attention_mask.squeeze(0)

            visual_features = self._load_visual_features(
                features_path=instance.features_path, modality=instance.modality
            )

            return EmmaDatasetItem(
                input_token_ids=input_encoding.input_ids.squeeze(0),
                text_attention_mask=input_encoding.attention_mask.squeeze(0),
                target_token_ids=target_token_ids,
                decoder_attention_mask=decoder_attention_mask,
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

        source_text = self._get_random_template_for_task(Task.captioning)

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        target_text = [caption.strip() for caption in instance.captions]

        visual_features = self._load_visual_features(
            features_path=instance.features_path, modality=instance.modality
        )

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=None,
            decoder_attention_mask=None,
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            raw_target=target_text,
            task=self._get_task_as_tensor(Task.captioning),
        )

    def _unpack_annotations(
        self,
    ) -> tuple[dict[int, dict[str, int]], int]:
        """Unpack the annotations from the db."""
        db_size = len(self.db)
        unpacked2packed: dict[int, dict[str, int]] = {}
        offset = 0
        dataset_size = 0
        with self.db:
            for index in range(db_size):
                instance_str: str = self.db[index]
                instance = CocoInstance.parse_raw(instance_str)
                for num_caption, _ in enumerate(instance.captions):
                    unpacked2packed[offset + index + num_caption] = {
                        "db_index": index,
                        "caption_index": num_caption,
                    }
                    dataset_size += 1
                offset += len(instance.captions) - 1
        return unpacked2packed, dataset_size
