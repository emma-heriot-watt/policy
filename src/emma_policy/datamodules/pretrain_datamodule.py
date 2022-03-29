import itertools
import os
from pathlib import Path
from typing import Optional, Union

from emma_datasets.common import Downloader, Settings
from emma_datasets.datamodels import DatasetSplit
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer

from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch, EmmaDatasetItem
from emma_policy.datamodules.pretrain_dataset import EmmaPretrainDataset
from emma_policy.datamodules.pretrain_instances import (
    PRETRAIN_DATASET_SPLITS,
    EnabledTasksHandler,
    EnabledTasksPerModality,
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
        batch_size: int = 8,
        model_name: str = "heriot-watt/emma-base",
        mlm_probability: float = 0.3,
        max_lang_tokens: Optional[int] = None,
        max_frames: int = 0,
        enabled_tasks: Optional[EnabledTasksPerModality] = None,
        tokenizer_truncation_side: str = "right",
    ) -> None:
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

        self._batch_size = batch_size
        self._num_workers = num_workers
        self.load_valid_data = load_valid_data
        self.max_lang_tokens = max_lang_tokens
        self.max_frames = max_frames
        self.tokenizer_truncation_side = tokenizer_truncation_side

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

        self._train_dataset = self._get_dataset_from_tasks(DatasetSplit.train)

        if self.load_valid_data:
            self._val_dataset = self._get_dataset_from_tasks(DatasetSplit.valid)

    def train_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate train dataloader."""
        return DataLoader(
            self._train_dataset,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate val dataloader."""
        return DataLoader(
            self._val_dataset,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

    def _get_dataset_from_tasks(
        self, dataset_split: DatasetSplit
    ) -> ConcatDataset[Optional[EmmaDatasetItem]]:
        """Iterate over all the enabled tasks and create a concatenated dataset."""
        all_datasets: list[EmmaPretrainDataset] = []

        for task in itertools.chain.from_iterable(self._enabled_tasks.values()):
            task_db_name = get_db_file_name(task, dataset_split)

            dataset = EmmaPretrainDataset(
                dataset_db_path=self._pretrain_db_dir_path.joinpath(task_db_name),
                tokenizer=self.tokenizer,
                mlm_probability=self.mlm_probability,
                max_frames=self.max_frames,
            )
            all_datasets.append(dataset)

        return ConcatDataset(all_datasets)

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
            for task, dataset_split in zip(
                itertools.chain.from_iterable(self._enabled_tasks.values()),
                itertools.cycle(PRETRAIN_DATASET_SPLITS),
            )
        ]

        all_urls = [self._remote_pretrain_db_dir + file_name for file_name in all_db_file_names]

        downloader = Downloader()
        downloader.download(all_urls, self._pretrain_db_dir_path)
