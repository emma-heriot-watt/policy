from pathlib import Path
from typing import Optional, Union

from emma_datasets.common import Settings, get_progress
from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from emma_policy.datamodules.pretrain_dataset import EmmaDatasetOutput, EmmaPretrainDataset
from emma_policy.datamodules.pretrain_instances import convert_instance_to_pretrain_instances


DEFAULT_DATASET_DB_PATH = Settings().paths.databases.joinpath("instances.db")


class EmmaPretrainDataModule(LightningDataModule):
    """DataModule to load data for the EMMA Pretraining Model."""

    def __init__(
        self,
        pretrain_instances_db_file: Union[str, Path],
        instances_db_file: Union[str, Path, None] = DEFAULT_DATASET_DB_PATH,
        force_prepare_data: bool = False,
        num_workers: int = 0,
        batch_size: int = 8,
    ) -> None:
        if isinstance(instances_db_file, str):
            instances_db_file = Path(instances_db_file)

        if isinstance(pretrain_instances_db_file, str):
            pretrain_instances_db_file = Path(pretrain_instances_db_file)

        self._instances_db_file = instances_db_file
        self._pretrain_instances_db_file = pretrain_instances_db_file

        self._instances_db_file_exists = (
            self._instances_db_file is not None and self._instances_db_file.exists()
        )

        if not self._pretrain_instances_db_file.exists() and not self._instances_db_file_exists:
            raise ValueError(
                "Both `instances_db_file` and `pretrain_instances_db_file` cannot be None. At least one MUST be provided."
            )

        self._force_prepare_data = force_prepare_data

        self._batch_size = batch_size
        self._num_workers = num_workers

    def prepare_data(self) -> None:
        """Prepare the DatasetDb for the pretraining.

        This will only create the pretraining instances db file if it does not already exist.
        """
        super().prepare_data()

        if not self._pretrain_instances_db_file.exists() or self._force_prepare_data:
            self._prepare_pretrain_instances_db()

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the dataloaders."""
        self._train_dataset = EmmaPretrainDataset(self._pretrain_instances_db_file)
        self._val_dataset = EmmaPretrainDataset(self._pretrain_instances_db_file)
        self._test_dataset = EmmaPretrainDataset(self._pretrain_instances_db_file)

    def train_dataloader(self) -> DataLoader[EmmaDatasetOutput]:
        """Generate train dataloader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> DataLoader[EmmaDatasetOutput]:
        """Generate val dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> DataLoader[EmmaDatasetOutput]:
        """Generate test dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def _prepare_pretrain_instances_db(self) -> None:
        instances_db = DatasetDb(self._instances_db_file)
        pretrain_instances_db = DatasetDb(self._pretrain_instances_db_file, readonly=False)
        progress = get_progress()

        get_instance_task_id = progress.add_task("Getting instances", total=len(instances_db))
        create_instances_task_id = progress.add_task(
            "Creating pretrain instances", total=float("inf")
        )

        with instances_db, pretrain_instances_db, progress:  # noqa: WPS316
            data_idx = 0

            for _, _, instance_str in instances_db:
                instance = Instance.parse_raw(instance_str)
                progress.advance(get_instance_task_id)

                current_instances = convert_instance_to_pretrain_instances(instance)

                for pretrain_instance in current_instances:
                    pretrain_instances_db[(data_idx, f"pretrain_{data_idx}")] = pretrain_instance

                    progress.advance(create_instances_task_id)
                    data_idx += 1
