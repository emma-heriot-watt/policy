import random
from collections import Counter
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from emma_datasets.constants.simbot.simbot import get_arena_definitions
from emma_datasets.datamodels.datasets.simbot import SimBotAction, SimBotInstructionInstance
from emma_datasets.datamodels.datasets.utils.simbot_utils import (
    get_object_from_action_object_metadata,
)
from emma_datasets.db import DatasetDb
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, PreTrainedTokenizer

from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.simbot_action_dataset import SimBotActionDataset


SimBotAction_SPECIAL_TOKENS = [
    "<stop>",
]


def prepare_action_tokenizer(
    model_name: str = "heriot-watt/emma-base",
    tokenizer_truncation_side: Literal["left", "right"] = "right",
    max_lang_tokens: Optional[int] = 64,
) -> PreTrainedTokenizer:
    """Add special tokens to tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": SimBotAction_SPECIAL_TOKENS}
    )  # doesn't add if they are already there
    tokenizer.truncation_side = tokenizer_truncation_side

    if max_lang_tokens:
        tokenizer.model_max_length = max_lang_tokens
    return tokenizer


class SimBotActionDataModule(LightningDataModule):
    """Data module to load SimBot actions for the EMMA Policy model."""

    def __init__(
        self,
        simbot_action_train_db_file: Union[str, Path],
        simbot_action_valid_db_file: Union[str, Path],
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
    ) -> None:
        super().__init__()
        if isinstance(simbot_action_train_db_file, str):
            simbot_action_train_db_file = Path(simbot_action_train_db_file)
        if isinstance(simbot_action_valid_db_file, str):
            simbot_action_valid_db_file = Path(simbot_action_valid_db_file)

        self._simbot_action_train_db_file = simbot_action_train_db_file
        self._simbot_action_valid_db_file = simbot_action_valid_db_file

        # Dataloader constraints
        self._max_lang_tokens = max_lang_tokens
        self._tokenizer_truncation_side = tokenizer_truncation_side
        self._num_workers = num_workers
        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._max_frames = max_frames
        self._weighted_sampling = weighted_sampling
        self._weight_temperature = weight_temperature
        self._subsample_perc = 0.5
        self._skip_goto_objects = {"Desk", "Table"}
        self._iou_threshold = iou_threshold

        # Model
        self._model_name = model_name

        arena_definitions = get_arena_definitions()
        self._object_assets_to_names = arena_definitions["asset_to_name"]
        self._image_width = arena_definitions["image_width"]
        self._image_height = arena_definitions["image_height"]

    def prepare_data(self) -> None:
        """Perform any preparation steps necessary before loading the data to the model."""
        super().prepare_data()

        AutoTokenizer.from_pretrained(self._model_name)

    def setup_tokenizer(self) -> PreTrainedTokenizer:
        """Add special tokens to tokenizer."""
        self._tokenizer = prepare_action_tokenizer(
            model_name=self._model_name,
            tokenizer_truncation_side=self._tokenizer_truncation_side,
            max_lang_tokens=self._max_lang_tokens,
        )
        return self._tokenizer

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the dataloaders."""
        self.setup_tokenizer()

        self._train_dataset = SimBotActionDataset(
            dataset_db_path=self._simbot_action_train_db_file,
            tokenizer=self._tokenizer,
            max_frames=self._max_frames,
            iou_threshold=self._iou_threshold,
        )

        self._training_sampler_weights: Optional[list[float]] = None

        self._valid_dataset = SimBotActionDataset(
            dataset_db_path=self._simbot_action_valid_db_file,
            tokenizer=self._tokenizer,
            max_frames=self._max_frames,
            iou_threshold=self._iou_threshold,
        )

        self._test_dataset = SimBotActionDataset(
            dataset_db_path=self._simbot_action_valid_db_file,
            tokenizer=self._tokenizer,
            max_frames=self._max_frames,
            iou_threshold=self._iou_threshold,
        )

    def train_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate train dataloader for SimBot action instances."""
        if self._weighted_sampling:
            self._training_sampler_weights = self._compute_sample_weights(
                self._simbot_action_train_db_file
            )
            return DataLoader(
                self._train_dataset,  # type: ignore[arg-type]
                batch_size=self._train_batch_size,
                num_workers=self._num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                sampler=WeightedRandomSampler(
                    self._training_sampler_weights, num_samples=len(self._train_dataset)
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
        """Generate valid dataloader for SimBot action instances."""
        return DataLoader(
            self._valid_dataset,  # type: ignore[arg-type]
            batch_size=self._val_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate test dataloader for SimBot action instances."""
        return DataLoader(
            self._test_dataset,  # type: ignore[arg-type]
            batch_size=self._val_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
        )

    def _compute_sample_weights(self, dataset_db: Path) -> list[float]:
        """Proportional temperature scaling to mitigate action type imbalance."""
        db = DatasetDb(dataset_db)
        # First pass through the dataset to get action type counts
        actions = []
        subsampled_weight: list[float] = []
        for _, _, instance_str in db:
            instance = SimBotInstructionInstance.parse_raw(instance_str)
            if self._skip_instance(instance.actions[-1]):
                subsampled_weight.append(0)
            else:
                subsampled_weight.append(1)
            actions.append(self._get_action_type(instance.actions[-1]))

        action_types = [
            action for weight, action in zip(subsampled_weight, actions) if weight == 1
        ]
        counts = Counter(actions)
        action_types = list(counts.keys())
        action_counts = np.array([counts[action] for action in action_types])
        probas = 1 / action_counts

        # Update the sampling probabilities through temperature scaling
        scaled_probas = probas ** (1 / self._weight_temperature)
        scaled_probas = scaled_probas / scaled_probas.sum()
        action_type_weights = dict(zip(action_types, scaled_probas))

        # Second pass to get the final weight of each sample
        data_weights = []
        for weight, action_type in zip(subsampled_weight, actions):
            data_weights.append(float(weight) * action_type_weights[action_type])
        return data_weights

    def _get_action_type(self, action: SimBotAction) -> str:
        if action.type != "Goto":
            return action.type
        if "officeRoom" in action.goto["object"]:
            return "Goto-Room"

        return "Goto-Object"

    def _skip_instance(self, action: SimBotAction) -> bool:
        """Randomly skip common action types."""
        action_type = self._get_action_type(action)
        # Check if action is Look Around or Goto object
        action_is_look_around = action_type == "Look" and action.look["direction"] == "Around"
        action_is_goto_object = action_type == "Goto-Object"
        if not (action_is_look_around or action_is_goto_object):
            return False
        # Probability of skipping the instance
        skip_instance = random.random() < self._subsample_perc
        # Skip Look Around
        if action_is_look_around:
            return skip_instance
        # Skip Goto Desk or Table
        target_object = get_object_from_action_object_metadata(
            action.goto["object"]["id"], self._object_assets_to_names
        )
        skip_target_object = target_object in self._skip_goto_objects
        return skip_target_object and skip_instance
