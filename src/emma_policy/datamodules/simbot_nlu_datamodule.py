from collections import Counter
from pathlib import Path
from random import shuffle
from typing import Literal, Optional, Union

from emma_datasets.datamodels import DatasetSplit
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, PreTrainedTokenizer

from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.simbot_nlu_dataset import SimBotNLUDataset, SimBotNLUIntents


SimBotNLU_SPECIAL_TOKENS = [
    "<act>",
    "<clarify>",
    "<decription>",
    "<location>",
    "<low_level>",
    "<disambiguation>",
    "<direction>",
    "<search>",
]


def prepare_nlu_tokenizer(
    model_name: str = "heriot-watt/emma-base",
    tokenizer_truncation_side: Literal["left", "right"] = "right",
    max_lang_tokens: Optional[int] = 64,
) -> PreTrainedTokenizer:
    """Add special tokens to tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": SimBotNLU_SPECIAL_TOKENS}
    )  # doesn't add if they are already there
    tokenizer.truncation_side = tokenizer_truncation_side

    if max_lang_tokens:
        tokenizer.model_max_length = max_lang_tokens
    return tokenizer


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
        balance_dataset: bool = False,
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

        self.balance_dataset = balance_dataset

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
            allow_paraphrasing=True,
        )

        if self.balance_dataset:
            self.balanced_num_samples = self._get_balanced_dataset_length(DatasetSplit.train)

        self._valid_dataset = SimBotNLUDataset(
            dataset_db_path=self._valid_db_file,
            tokenizer=self.tokenizer,
            is_train=False,
            allow_paraphrasing=False,
        )

        self._test_dataset = SimBotNLUDataset(
            dataset_db_path=self._test_db_file,
            tokenizer=self.tokenizer,
            is_train=False,
            allow_paraphrasing=False,
        )

    def train_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate train dataloader for SimBot NLU instances."""
        if self.balance_dataset:
            # Resample at the beginning of each epoch.
            train_dataset = self._get_balanced_train_datasets()
        else:
            train_dataset = self._train_dataset

        return DataLoader(
            train_dataset,  # type: ignore[arg-type]
            batch_size=self._train_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=True,
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

    def _get_balanced_dataset_length(self, dataset_split: DatasetSplit) -> int:
        """Balance the number of samples from datasets of different tasks."""
        if dataset_split != DatasetSplit.train:
            raise AssertionError("Balancing only supported for training datasets.")
        dataset_lengths = list(Counter(self._train_dataset.data_intents).values())
        return min(dataset_lengths)

    def _get_balanced_train_datasets(
        self,
    ) -> SimBotNLUDataset:
        """Create the dataset for each enabled task."""
        train_indices = []
        for nlu_intent in SimBotNLUIntents:
            class_indices = [
                index
                for index, intent in enumerate(self._train_dataset.data_intents)
                if intent == nlu_intent
            ]
            shuffle(class_indices)
            train_indices.extend(class_indices[: self.balanced_num_samples])
        dataset = Subset(self._train_dataset, train_indices)
        return dataset  # type: ignore[return-value]
