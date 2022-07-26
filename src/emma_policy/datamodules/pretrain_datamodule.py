import itertools
import os
from pathlib import Path
from typing import Optional, Union

import torch
from emma_datasets.common import Downloader, Settings
from emma_datasets.datamodels import DatasetSplit
from emma_datasets.db import DatasetDb
from overrides import overrides
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Subset
from transformers import AutoTokenizer

from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch, EmmaDatasetItem
from emma_policy.datamodules.pretrain_dataset import EmmaPretrainDataset
from emma_policy.datamodules.pretrain_instances import (
    PRETRAIN_DATASET_SPLITS,
    EnabledTasksHandler,
    EnabledTasksPerModality,
    Task,
    get_db_file_name,
)


DEFAULT_DB_PATH = Settings().paths.databases


class EmmaPretrainDataModule(LightningDataModule):  # noqa: WPS230
    """DataModule to load data for the EMMA Pretraining Model."""

    def __init__(
        self,
        pretrain_db_dir_path: Union[str, Path] = DEFAULT_DB_PATH,
        remote_pretrain_db_dir: str = "s3://emma-simbot/db/",
        load_valid_data: bool = False,
        num_workers: int = 0,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        model_name: str = "heriot-watt/emma-base",
        mlm_probability: float = 0.3,
        max_lang_tokens: Optional[int] = None,
        max_frames: int = 0,
        enabled_tasks: Optional[EnabledTasksPerModality] = None,
        tokenizer_truncation_side: str = "right",
        balance_datasets: bool = False,
        balancing_ratio: int = 2,
        shuffle_objects: bool = False,
        propotional_task_sampling: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.mlm_probability = mlm_probability

        self._enabled_tasks = (
            EnabledTasksHandler.get_default_enabled_tasks_per_modality()
            if enabled_tasks is None
            else EnabledTasksHandler.process_tasks_per_modality(enabled_tasks)
        )

        if isinstance(pretrain_db_dir_path, str):
            pretrain_db_dir_path = Path(pretrain_db_dir_path)

        if not pretrain_db_dir_path.is_dir():
            raise AssertionError("`pretrain_db_dir_path` needs to point to a directory.")

        self._pretrain_db_dir_path = pretrain_db_dir_path
        self._remote_pretrain_db_dir = remote_pretrain_db_dir

        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._num_workers = num_workers
        self.load_valid_data = load_valid_data
        self.max_lang_tokens = max_lang_tokens
        self.max_frames = max_frames
        self.tokenizer_truncation_side = tokenizer_truncation_side
        self.balance_datasets = self._verify_balance_datasets(balance_datasets)
        self.balanced_num_samples = -1
        self.balancing_ratio = balancing_ratio
        self.shuffle_objects = shuffle_objects
        self.propotional_task_sampling = propotional_task_sampling

    def prepare_data(self) -> None:
        """Download the pretrain DBs if necessary.

        This will NOT download them if running the code using pytest.
        """
        super().prepare_data()

        # Make the directory for the pretrain DBs if it does not already exist
        if not self._pretrain_db_dir_path.exists():
            self._pretrain_db_dir_path.mkdir(parents=True, exist_ok=True)

        # Only download the pretrain DBs if not running tests
        if os.getenv("RUNNING_TESTS", "0") != "1":
            self._download_pretrain_dbs()

        # make sure to trigger the tokenizer download on the main process
        AutoTokenizer.from_pretrained(self.model_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the dataloaders."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.truncation_side = self.tokenizer_truncation_side
        if self.max_lang_tokens:
            self.tokenizer.model_max_length = self.max_lang_tokens

        if self.balance_datasets:
            self.balanced_num_samples = self._get_balanced_dataset_length(DatasetSplit.train)
        self._train_dataset = self._get_datasets_from_tasks(DatasetSplit.train)

        if self.load_valid_data:
            self._val_dataset = self._get_concatenated_dataset_from_tasks(DatasetSplit.valid)

    @overrides(check_signature=False)
    def train_dataloader(
        self,
    ) -> Union[dict[Task, DataLoader[EmmaDatasetBatch]], DataLoader[EmmaDatasetBatch]]:
        """Generate train dataloader."""
        if self.balance_datasets:
            # Resample at the beginning of each epoch.
            train_dataset = self._get_datasets_from_tasks(DatasetSplit.train)
        else:
            train_dataset = self._train_dataset

        if self.propotional_task_sampling:
            return {
                task: DataLoader(
                    dataset,  # type: ignore[arg-type]
                    batch_size=self._train_batch_size,
                    num_workers=self._num_workers,
                    collate_fn=collate_fn,
                    shuffle=True,
                    pin_memory=True,
                )
                for task, dataset in train_dataset.items()
            }

        combined_dataset: ConcatDataset[Optional[EmmaDatasetItem]] = ConcatDataset(
            train_dataset.values()
        )
        return DataLoader(
            combined_dataset,  # type: ignore[arg-type]
            batch_size=self._train_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate val dataloader."""
        return DataLoader(
            self._val_dataset,  # type: ignore[arg-type]
            batch_size=self._val_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

    def _get_datasets_from_tasks(
        self,
        dataset_split: DatasetSplit,
    ) -> dict[Task, EmmaPretrainDataset]:
        """Create the dataset for each enabled task."""
        all_datasets: dict[Task, EmmaPretrainDataset] = {}

        for task in itertools.chain.from_iterable(self._enabled_tasks.values()):
            task_db_name = get_db_file_name(task, dataset_split)

            dataset = EmmaPretrainDataset(
                dataset_db_path=self._pretrain_db_dir_path.joinpath(task_db_name),
                tokenizer=self.tokenizer,
                mlm_probability=self.mlm_probability,
                max_frames=self.max_frames,
                shuffle_objects=self.shuffle_objects,
            )
            if self.balance_datasets and dataset_split == DatasetSplit.train:
                indices = torch.randperm(len(dataset))[
                    : self.balancing_ratio * self.balanced_num_samples
                ]
                dataset = Subset(dataset, indices.tolist())  # type: ignore[assignment]

            all_datasets[task] = dataset

        return all_datasets

    def _get_concatenated_dataset_from_tasks(
        self,
        dataset_split: DatasetSplit,
    ) -> ConcatDataset[Optional[EmmaDatasetItem]]:
        """Iterate over all the enabled tasks and create a concatenated dataset."""
        all_datasets = self._get_datasets_from_tasks(dataset_split)
        return ConcatDataset(all_datasets.values())

    def _get_balanced_dataset_length(self, dataset_split: DatasetSplit) -> int:
        """Balance the number of samples from datasets of different tasks."""
        if dataset_split != DatasetSplit.train:
            raise AssertionError("Balancing only supported for training datasets.")
        dataset_lengths: list[int] = []
        for task in itertools.chain.from_iterable(self._enabled_tasks.values()):
            task_db_name = get_db_file_name(task, dataset_split)
            dataset_db_path = self._pretrain_db_dir_path.joinpath(task_db_name)

            dataset_lengths.append(len(DatasetDb(dataset_db_path)))

        return min(dataset_lengths)

    def _verify_balance_datasets(self, balance_datasets: bool) -> bool:
        """Make sure that at least two tasks are enabled for balancing."""
        tasks = set(itertools.chain.from_iterable(self._enabled_tasks.values()))
        if len(tasks) == 1:
            balance_datasets = False
        return balance_datasets

    def _download_pretrain_dbs(self) -> None:
        """Download all the pretrain DBs from S3.

        Get all the file names for every DB that needs to be downloaded, convert into a URL and
        then download them all in one go.

        The `Downloader` will skip the file if it already exists --- determined by making sure the
        local file is not smaller than the requested file.
        """
        if not self._remote_pretrain_db_dir.endswith("/"):
            self._remote_pretrain_db_dir = f"{self._remote_pretrain_db_dir}/"

        all_db_file_names = [
            get_db_file_name(task, dataset_split)
            for task, dataset_split in itertools.product(
                itertools.chain.from_iterable(self._enabled_tasks.values()),
                PRETRAIN_DATASET_SPLITS,
            )
        ]

        all_urls = [self._remote_pretrain_db_dir + file_name for file_name in all_db_file_names]

        downloader = Downloader()
        downloader.download(all_urls, self._pretrain_db_dir_path)
