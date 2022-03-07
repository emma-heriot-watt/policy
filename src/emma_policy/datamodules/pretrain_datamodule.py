from pathlib import Path
from typing import Optional, Union

from emma_datasets.common import Settings
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem
from emma_policy.datamodules.pretrain_dataset import EmmaPretrainDataset
from emma_policy.datamodules.pretrain_instances import (
    DEFAULT_COCO_SPLITS_PATH,
    PreparePretrainInstancesDb,
    load_ref_coco_images,
)
from emma_policy.datamodules.pretrain_instances.datamodels import (
    EnabledTasksHandler,
    EnabledTasksPerModality,
)


DEFAULT_DATASET_DB_PATH = Settings().paths.databases.joinpath("instances.db")


class EmmaPretrainDataModule(LightningDataModule):  # noqa: WPS230
    """DataModule to load data for the EMMA Pretraining Model."""

    def __init__(
        self,
        pretrain_train_db_file: Union[str, Path],
        pretrain_valid_db_file: Union[str, Path],
        instances_db_file: Union[str, Path] = DEFAULT_DATASET_DB_PATH,
        force_prepare_data: bool = False,
        load_valid_data: bool = False,
        num_workers: int = 0,
        prepare_data_num_workers: int = 0,
        batch_size: int = 8,
        coco_split_path: Union[str, Path] = DEFAULT_COCO_SPLITS_PATH,
        model_name: str = "heriot-watt/emma-base",
        mlm_probability: float = 0.3,
        max_lang_tokens: Optional[int] = None,
        max_frames: Optional[int] = None,
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

        if isinstance(instances_db_file, str):
            instances_db_file = Path(instances_db_file)

        if isinstance(pretrain_train_db_file, str):
            pretrain_train_db_file = Path(pretrain_train_db_file)

        if isinstance(pretrain_valid_db_file, str):
            pretrain_valid_db_file = Path(pretrain_valid_db_file)

        if isinstance(coco_split_path, str):
            coco_split_path = Path(coco_split_path)

        self._instances_db_file = instances_db_file
        self._pretrain_train_db_file = pretrain_train_db_file
        self._pretrain_valid_db_file = pretrain_valid_db_file

        self._instances_db_file_exists = (
            self._instances_db_file is not None and self._instances_db_file.exists()
        )

        no_pretrain_db_files = (
            not self._pretrain_train_db_file.exists() or not self._pretrain_valid_db_file.exists()
        )

        if no_pretrain_db_files and not self._instances_db_file_exists:
            raise ValueError(
                "Both `instances_db_file` and `pretrain_*_db_file` cannot be None. At least one MUST be provided."
            )

        self._force_prepare_data = force_prepare_data
        self._prepare_data_num_workers = prepare_data_num_workers

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._coco_ref_images = load_ref_coco_images(coco_split_path)
        self.load_valid_data = load_valid_data
        self.max_lang_tokens = max_lang_tokens
        self.max_frames = max_frames
        self.tokenizer_truncation_side = tokenizer_truncation_side

    def prepare_data(self) -> None:
        """Prepare the DatasetDb for the pretraining.

        This will only create the pretraining instances db file if it does not already exist.
        """
        super().prepare_data()

        if not self._pretrain_train_db_file.exists() or self._force_prepare_data:
            self._prepare_pretrain_instances_db()

        # make sure to trigger the tokenizer download on the main process
        AutoTokenizer.from_pretrained(self.model_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the dataloaders."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.truncation_side = self.tokenizer_truncation_side
        if self.max_lang_tokens:
            self.tokenizer.model_max_length = self.max_lang_tokens

        self._train_dataset = EmmaPretrainDataset(
            dataset_db_path=self._pretrain_train_db_file,
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
            max_frames=self.max_frames,
        )

        if self.load_valid_data:
            self._val_dataset = EmmaPretrainDataset(
                dataset_db_path=self._pretrain_valid_db_file,
                tokenizer=self.tokenizer,
                mlm_probability=self.mlm_probability,
                max_frames=self.max_frames,
            )

    def train_dataloader(self) -> DataLoader[Optional[EmmaDatasetItem]]:
        """Generate train dataloader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[Optional[EmmaDatasetItem]]:
        """Generate val dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

    def _prepare_pretrain_instances_db(self) -> None:
        loader_batch_size = 512

        if self._prepare_data_num_workers > loader_batch_size:
            raise AssertionError("Ensure the `num_workers` is less than the `loader_batch_size`")

        preparer = PreparePretrainInstancesDb(
            instances_db_file_path=self._instances_db_file,
            coco_ref_images=self._coco_ref_images,
            train_db_file_path=self._pretrain_train_db_file,
            valid_db_file_path=self._pretrain_valid_db_file if self.load_valid_data else None,
            loader_batch_size=loader_batch_size,
            loader_num_workers=self._prepare_data_num_workers,
            enabled_tasks=self._enabled_tasks,
        )
        preparer.run()
