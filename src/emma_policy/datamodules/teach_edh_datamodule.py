from pathlib import Path
from typing import Literal, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer

from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.teach_edh_dataset import TeachEdhDataset


class TeachEdhDataModule(LightningDataModule):
    """Data module to load EDH instances for the EMMA Policy model."""

    def __init__(
        self,
        teach_edh_train_db_file: Union[str, Path],
        teach_edh_valid_seen_db_file: Union[str, Path],
        teach_edh_valid_unseen_db_file: Union[str, Path],
        load_valid_data_split: Optional[Literal["seen", "unseen", "both"]] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        model_name: str = "heriot-watt/emma-base",
        max_lang_tokens: Optional[int] = None,
        max_frames: int = 0,
        tokenizer_truncation_side: Literal["left", "right"] = "right",
    ) -> None:
        if isinstance(teach_edh_train_db_file, str):
            teach_edh_train_db_file = Path(teach_edh_train_db_file)
        if isinstance(teach_edh_valid_seen_db_file, str):
            teach_edh_valid_seen_db_file = Path(teach_edh_valid_seen_db_file)
        if isinstance(teach_edh_valid_unseen_db_file, str):
            teach_edh_valid_unseen_db_file = Path(teach_edh_valid_unseen_db_file)

        self._teach_edh_train_db_file = teach_edh_train_db_file
        self._teach_edh_valid_seen_db_file = teach_edh_valid_seen_db_file
        self._teach_edh_valid_unseen_db_file = teach_edh_valid_unseen_db_file

        # Preparation
        self._load_valid_data_split = load_valid_data_split

        # Dataloader constraints
        self._max_lang_tokens = max_lang_tokens
        self._max_frames = max_frames
        self._tokenizer_truncation_side = tokenizer_truncation_side
        self._num_workers = num_workers
        self._batch_size = batch_size

        # Model
        self._model_name = model_name

    def prepare_data(self) -> None:
        """Perform any preparation steps necessary before loading the data to the model."""
        super().prepare_data()

        AutoTokenizer.from_pretrained(self._model_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the dataloaders."""
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._tokenizer.truncation_side = self._tokenizer_truncation_side

        if self._max_lang_tokens:
            self._tokenizer.model_max_length = self._max_lang_tokens

        self._train_dataset = TeachEdhDataset(
            dataset_db_path=self._teach_edh_train_db_file,
            tokenizer=self._tokenizer,
            max_frames=self._max_frames,
        )

        if self._load_valid_data_split:
            self._valid_seen_dataset = TeachEdhDataset(
                dataset_db_path=self._teach_edh_valid_seen_db_file,
                tokenizer=self._tokenizer,
                max_frames=self._max_frames,
            )
            self._valid_unseen_dataset = TeachEdhDataset(
                dataset_db_path=self._teach_edh_valid_unseen_db_file,
                tokenizer=self._tokenizer,
                max_frames=self._max_frames,
            )

    def train_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate train dataloader for TEACh EDH instances."""
        return DataLoader(
            self._train_dataset,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate validation dataloader for the TEACh EDH instances.

        Default to returning the valid seen dataset because there needs to be a return else it will
        causes exceptions down the line.
        """
        if self._load_valid_data_split == "unseen":
            return DataLoader(
                self._valid_unseen_dataset,  # type: ignore[arg-type]
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                collate_fn=collate_fn,
                shuffle=False,
            )

        if self._load_valid_data_split == "both":
            return DataLoader(
                ConcatDataset([self._valid_seen_dataset, self._valid_unseen_dataset]),
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                collate_fn=collate_fn,
                shuffle=False,
            )

        return DataLoader(
            self._valid_seen_dataset,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )
