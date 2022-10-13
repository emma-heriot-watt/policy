from collections import Counter
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from emma_datasets.datamodels.datasets.simbot import SimBotInstructionInstance
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

        # Model
        self._model_name = model_name

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
        )
        self._training_sampler_weights = None
        if self._weighted_sampling:
            self._training_sampler_weights = self._compute_sample_weights(
                self._simbot_action_train_db_file
            )

        self._valid_dataset = SimBotActionDataset(
            dataset_db_path=self._simbot_action_valid_db_file,
            tokenizer=self._tokenizer,
            max_frames=self._max_frames,
        )

        self._test_dataset = SimBotActionDataset(
            dataset_db_path=self._simbot_action_valid_db_file,
            tokenizer=self._tokenizer,
            max_frames=self._max_frames,
        )

    def train_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate train dataloader for SimBot action instances."""
        if self._training_sampler_weights is not None:
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
        for _, _, instance_str in db:
            instance = SimBotInstructionInstance.parse_raw(instance_str)
            actions.append(instance.actions[-1].type)

        counts = Counter(actions)
        action_types = list(counts.keys())
        action_counts = np.array([counts[action] for action in action_types])
        probas = 1 / action_counts

        # Update the sampling probabilities through temperature scaling
        scaled_probas = probas ** (1 / self._weight_temperature)
        scaled_probas = scaled_probas / scaled_probas.sum()
        action_type_weights = dict(zip(action_types, scaled_probas))

        # Second pass to get the weight of each sample
        data_weights = []
        for _, _, instance_str in db:  # noqa: WPS440
            instance = SimBotInstructionInstance.parse_raw(instance_str)
            data_weights.append(action_type_weights[instance.actions[-1].type])
        return data_weights
