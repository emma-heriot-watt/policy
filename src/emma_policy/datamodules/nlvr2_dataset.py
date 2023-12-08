from pathlib import Path

from emma_datasets.datamodels.datasets.nlvr import NlvrInstance
from overrides import overrides
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.utils import get_logger


logger = get_logger(__name__)


class NLVR2Dataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for NLVR2.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_frames: int = 0,
        use_task_prefix: bool = False,
    ) -> None:
        super().__init__(
            dataset_db_path=dataset_db_path, tokenizer=tokenizer, max_frames=max_frames
        )

        self.dataset_size = len(self.db)
        self._use_task_prefix = use_task_prefix

    @overrides(check_signature=False)
    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return self.dataset_size

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get the NLVR2 instance at the given index as an instance of `EmmaDatasetItem`."""
        instance = NlvrInstance.parse_raw(self.db[index])
        return self.nlvr2(instance)

    def nlvr2(self, instance: NlvrInstance) -> EmmaDatasetItem:
        """Process the instance for the NLVR2 task."""
        if instance.image_ids is None:
            raise AssertionError(
                "image_ids for this instance must exist. Make sure this instance is connected to the right task!"
            )
        if self._use_task_prefix:
            source_text = self._get_random_template_for_task(Task.itm).format(
                statement=instance.sentence
            )
        else:
            source_text = instance.sentence

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        target_text = instance.label.rstrip()

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
            task=self._get_task_as_tensor(Task.itm),
        )
