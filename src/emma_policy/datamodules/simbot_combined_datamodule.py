from pathlib import Path
from typing import Literal, Optional, Union

from emma_datasets.constants.simbot.simbot import get_arena_definitions
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    SimBotAction,
    SimBotInstructionInstance,
)
from emma_datasets.db import DatasetDb
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch, EmmaDatasetItem
from emma_policy.datamodules.simbot_action_dataset import SimBotActionDataset
from emma_policy.datamodules.simbot_nlu_dataset import SimBotNLUDataset, SimBotNLUIntents
from emma_policy.utils import DistributedWeightedSampler, compute_weights


SimBotAction_SPECIAL_TOKENS = ["<stop>"]


def prepare_combined_tokenizer(
    model_name: str = "heriot-watt/emma-base",
    tokenizer_truncation_side: Literal["left", "right"] = "right",
    max_lang_tokens: Optional[int] = 64,
) -> PreTrainedTokenizer:
    """Add special tokens to tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # vad special tokens
    vad_special_tokens = [intent.value for intent in SimBotNLUIntents if intent.is_special_token]
    action_special_tokens = SimBotAction_SPECIAL_TOKENS
    combined_special_tokens = vad_special_tokens + action_special_tokens

    tokenizer.add_special_tokens(
        {"additional_special_tokens": combined_special_tokens}
    )  # doesn't add if they are already there
    tokenizer.truncation_side = tokenizer_truncation_side

    if max_lang_tokens:
        tokenizer.model_max_length = max_lang_tokens
    return tokenizer


class SimBotCombinedDataModule(LightningDataModule):
    """Data module to load SimBot actions for the EMMA Policy model."""

    def __init__(
        self,
        simbot_action_train_db_file: Union[str, Path],
        simbot_action_valid_db_file: Union[str, Path],
        simbot_vad_train_db_file: Union[str, Path],
        simbot_vad_valid_db_file: Union[str, Path],
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        model_name: str = "heriot-watt/emma-base",
        max_lang_tokens: Optional[int] = None,
        max_frames: int = 15,
        tokenizer_truncation_side: Literal["left", "right"] = "right",
        weighted_sampling: bool = True,
        weight_temperature: float = 1.3,
        iou_threshold: float = 0.5,
        skip_common_instances: bool = False,
        shuffle_objects: bool = False,
    ) -> None:
        super().__init__()
        self._simbot_action_train_db_file = Path(simbot_action_train_db_file)
        self._simbot_action_valid_db_file = Path(simbot_action_valid_db_file)

        self._simbot_vad_train_db_file = Path(simbot_vad_train_db_file)
        self._simbot_vad_valid_db_file = Path(simbot_vad_valid_db_file)

        # Dataset config
        self._max_lang_tokens = max_lang_tokens
        self._tokenizer_truncation_side = tokenizer_truncation_side
        self._max_frames = max_frames
        self._iou_threshold = iou_threshold
        self._shuffle_objects = shuffle_objects

        # Dataloader constraints
        self._num_workers = num_workers
        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._weighted_sampling = weighted_sampling
        self._weight_temperature = weight_temperature

        # Model
        self._model_name = model_name
        arena_definitions = get_arena_definitions()
        self._object_assets_to_names = arena_definitions["asset_to_label"]
        self._image_width = arena_definitions["image_width"]
        self._image_height = arena_definitions["image_height"]

    def prepare_data(self) -> None:
        """Perform any preparation steps necessary before loading the data to the model."""
        super().prepare_data()

        AutoTokenizer.from_pretrained(self._model_name)

    def setup_tokenizer(self) -> PreTrainedTokenizer:
        """Add special tokens to tokenizer."""
        self._tokenizer = prepare_combined_tokenizer(
            model_name=self._model_name,
            tokenizer_truncation_side=self._tokenizer_truncation_side,
            max_lang_tokens=self._max_lang_tokens,
        )
        return self._tokenizer

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the dataloaders."""
        self.setup_tokenizer()

        train_action_dataset = SimBotActionDataset(
            dataset_db_path=self._simbot_action_train_db_file,
            tokenizer=self._tokenizer,
            max_frames=self._max_frames,
            iou_threshold=self._iou_threshold,
            shuffle_objects=self._shuffle_objects,
            allow_paraphrasing=True,
        )

        # Train
        train_vad_dataset = SimBotNLUDataset(
            dataset_db_path=self._simbot_vad_train_db_file,
            tokenizer=self._tokenizer,
            is_train=True,
            shuffle_objects=self._shuffle_objects,
        )

        self._train_dataset: ConcatDataset[Optional[EmmaDatasetItem]] = ConcatDataset(
            [train_action_dataset, train_vad_dataset]
        )

        self._vad_data_intents = train_vad_dataset.data_intents

        # Validation
        valid_action_dataset = SimBotActionDataset(
            dataset_db_path=self._simbot_action_valid_db_file,
            tokenizer=self._tokenizer,
            max_frames=self._max_frames,
            iou_threshold=self._iou_threshold,
            allow_paraphrasing=True,
        )

        valid_vad_dataset = SimBotNLUDataset(
            dataset_db_path=self._simbot_vad_valid_db_file,
            tokenizer=self._tokenizer,
            is_train=False,
        )

        self._valid_dataset: ConcatDataset[Optional[EmmaDatasetItem]] = ConcatDataset(
            [valid_action_dataset, valid_vad_dataset]
        )

        # Test
        test_action_dataset = SimBotActionDataset(
            dataset_db_path=self._simbot_action_valid_db_file,
            tokenizer=self._tokenizer,
            max_frames=self._max_frames,
            iou_threshold=self._iou_threshold,
            allow_paraphrasing=True,
        )

        test_vad_dataset = SimBotNLUDataset(
            dataset_db_path=self._simbot_vad_valid_db_file,
            tokenizer=self._tokenizer,
            is_train=False,
        )

        self._test_dataset: ConcatDataset[Optional[EmmaDatasetItem]] = ConcatDataset(
            [test_action_dataset, test_vad_dataset]
        )

    def train_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate train dataloader for SimBot instances."""
        if self._weighted_sampling:
            training_sampler_weights = self._compute_sample_weights()
            return DataLoader(
                self._train_dataset,  # type: ignore[arg-type]
                batch_size=self._train_batch_size,
                num_workers=self._num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                sampler=DistributedWeightedSampler(
                    training_sampler_weights, total_size=len(self._train_dataset)
                ),
            )
        return DataLoader(
            self._train_dataset,  # type: ignore[arg-type]
            batch_size=self._train_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate valid dataloader for SimBot instances."""
        return DataLoader(
            self._valid_dataset,  # type: ignore[arg-type]
            batch_size=self._val_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate test dataloader for SimBot instances."""
        return DataLoader(
            self._test_dataset,  # type: ignore[arg-type]
            batch_size=self._val_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=False,
        )

    def _compute_sample_weights(self) -> list[float]:
        """Proportional temperature scaling to mitigate action type imbalance."""
        action_db = DatasetDb(self._simbot_action_train_db_file)
        # First pass through the dataset to get action type counts
        actions: list[Union[str, SimBotNLUIntents]] = []
        for _, _, instance_str in action_db:
            instance = SimBotInstructionInstance.parse_raw(instance_str)
            actions.append(self._get_action_type(instance.actions[-1]))

        actions.extend(self._vad_data_intents)
        data_weights = compute_weights(actions, temperature=self._weight_temperature)
        return data_weights

    def _get_action_type(self, action: SimBotAction) -> str:
        if action.type != "Goto":
            return action.type
        if "officeRoom" in action.goto["object"]:
            return "Goto-Room"

        return "Goto-Object"
