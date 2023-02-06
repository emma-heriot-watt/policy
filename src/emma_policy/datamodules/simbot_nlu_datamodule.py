from pathlib import Path
from typing import Literal, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.simbot_nlu_dataset import SimBotNLUDataset, SimBotNLUIntents
from emma_policy.utils import DistributedWeightedSampler, compute_weights


def prepare_nlu_tokenizer(
    model_name: str = "heriot-watt/emma-base",
    tokenizer_truncation_side: Literal["left", "right"] = "right",
    max_lang_tokens: Optional[int] = 64,
) -> PreTrainedTokenizer:
    """Add special tokens to tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = [intent.value for intent in SimBotNLUIntents if intent.is_special_token]
    # doesn't add if they are already there
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    tokenizer.truncation_side = tokenizer_truncation_side

    if max_lang_tokens:
        tokenizer.model_max_length = max_lang_tokens
    return tokenizer


def get_nlu_classes() -> list[str]:
    """Get the NLU classes."""
    return [intent.name for intent in SimBotNLUIntents if intent.is_nlu_output]


class SimBotNLUDataModule(LightningDataModule):
    """Data module to load SimBot instructions for the EMMA NLU model."""

    def __init__(
        self,
        train_db_file: Union[str, Path],
        valid_db_file: Union[str, Path],
        test_db_file: Union[str, Path],
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        model_name: str = "heriot-watt/emma-base",
        max_lang_tokens: Optional[int] = None,
        tokenizer_truncation_side: Literal["left", "right"] = "right",
        weighted_sampling: bool = True,
        weight_temperature: float = 1.5,
    ) -> None:
        super().__init__()
        if isinstance(train_db_file, str):
            train_db_file = Path(train_db_file)
        if isinstance(valid_db_file, str):
            valid_db_file = Path(valid_db_file)
        if isinstance(test_db_file, str):
            test_db_file = Path(test_db_file)

        self._train_db_file = train_db_file
        self._valid_db_file = valid_db_file
        self._test_db_file = test_db_file

        # Dataloader constraints
        self._max_lang_tokens = max_lang_tokens
        self.tokenizer_truncation_side = tokenizer_truncation_side

        self._num_workers = num_workers
        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size

        # Model
        self._model_name = model_name

        self._weighted_sampling = weighted_sampling
        self._weight_temperature = weight_temperature

    def prepare_data(self) -> None:
        """Perform any preparation steps necessary before loading the data to the model."""
        super().prepare_data()

        AutoTokenizer.from_pretrained(self._model_name)

    def setup_tokenizer(self) -> PreTrainedTokenizer:
        """Add special tokens to tokenizer."""
        self.tokenizer = prepare_nlu_tokenizer(
            model_name=self._model_name,
            tokenizer_truncation_side=self.tokenizer_truncation_side,
            max_lang_tokens=self._max_lang_tokens,
        )
        return self.tokenizer

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the dataloaders."""
        self.setup_tokenizer()

        self._train_dataset = SimBotNLUDataset(
            dataset_db_path=self._train_db_file,
            tokenizer=self.tokenizer,
            is_train=True,
            shuffle_objects=True,
        )

        self._valid_dataset = SimBotNLUDataset(
            dataset_db_path=self._valid_db_file,
            tokenizer=self.tokenizer,
            is_train=False,
        )

        self._test_dataset = SimBotNLUDataset(
            dataset_db_path=self._test_db_file,
            tokenizer=self.tokenizer,
            is_train=False,
        )

    def train_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate train dataloader for SimBot NLU instances."""
        if self._weighted_sampling:
            training_sampler_weights = compute_weights(
                self._train_dataset.data_intents,
                temperature=self._weight_temperature,
            )
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
        """Generate valid dataloader for SimBot NLU instances."""
        return DataLoader(
            self._valid_dataset,  # type: ignore[arg-type]
            batch_size=self._val_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate test dataloader for SimBot NLU instances."""
        return DataLoader(
            self._test_dataset,  # type: ignore[arg-type]
            batch_size=self._val_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )
