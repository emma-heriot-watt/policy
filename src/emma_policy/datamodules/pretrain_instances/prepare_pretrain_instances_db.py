import math
from collections import Counter
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import torch
from emma_datasets.common import get_progress
from emma_datasets.datamodels import Instance, MediaType
from emma_datasets.db import DatasetDb, JsonStorage
from torch.utils.data import DataLoader, IterableDataset

from emma_policy.datamodules.pretrain_instances import convert_instance_to_pretrain_instances
from emma_policy.datamodules.pretrain_instances.datamodels import EnabledTasksHandler, Task
from emma_policy.datamodules.pretrain_instances.is_train_instance import is_train_instance


class DatasetDbReaderReturn(NamedTuple):
    """Return tuple for the DatasetDbReader."""

    source_instance_idx: int
    pretrain_instance: bytes
    is_train: bool


class IterableDatasetDbReader(IterableDataset[DatasetDbReaderReturn]):
    """Read the entire DatasetDb using the power of pytorch.

    The DatasetDb is not opened until the iterator is called.
    """

    def __init__(
        self,
        db_path: Path,
        enabled_tasks: dict[MediaType, set[Task]],
    ) -> None:
        db = DatasetDb(db_path, readonly=True)

        self.db: Optional[DatasetDb] = None
        self.db_path = db_path
        self.start = 0
        self.end = len(db)

        self._storage = JsonStorage()
        self._enabled_tasks = enabled_tasks

    def __iter__(self) -> Iterator[DatasetDbReaderReturn]:
        """Iterate over the entire DatasetDb.

        If the `DataLoader` is using multiple workers, then the indexes to process are distributed
        across the workers.
        """
        if self.db is None:
            self.db = DatasetDb(self.db_path, readonly=True)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            iter_start = self.start
            iter_end = self.end

        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        for data_idx in range(iter_start, iter_end):
            data = self.db[data_idx]

            instance = Instance.parse_raw(data)
            is_train = is_train_instance(instance)

            pretrain_instance_iterator = convert_instance_to_pretrain_instances(
                instance=instance,
                enabled_tasks=self._enabled_tasks[instance.modality],
            )

            yield from (
                DatasetDbReaderReturn(
                    source_instance_idx=data_idx,
                    pretrain_instance=self._storage.compress(pretrain_instance),
                    is_train=is_train,
                )
                for pretrain_instance in pretrain_instance_iterator
            )


def _loader_collate_fn(batch: list[DatasetDbReaderReturn]) -> list[DatasetDbReaderReturn]:
    return batch


class PreparePretrainInstancesDb:
    """Prepare the pretraining data as instance db's.

    To run the preparation process, call `.run()` after initialising.

    This uses the power of pytorch to work fast, but it might still take a while if you
    have a _lot_ of instances.

    Args:
        instances_db_file_path (Path): Path to the DB of _all_ instances.
        coco_ref_images (CocoRefImages): Reference images for the coco dataset to determine whether
            an instance is for the training or validation set.
        train_db_file_path (Path): Output path to the training db file.
        valid_db_file_path (Optional[Path]): Output path to the validation db file. If left as
            `None`, the valid db will not be created. Defaults to None.
        loader_batch_size (int): Batch size for the torch DataLoader. Defaults to 48.
        loader_num_workers (int): Number of workers for the torch DataLoader. Defaults to 24.
        write_db_batch_size (int): Total cache size for the output dbs before writing to disk.
            If you are running out of memory, you should make this value smaller. Alternative, if
            you are finding it slowing down too much, set this number to be much larger. Defaults
            to 300000.
        example_id_prefix (str): Example ID prefix used when saving examples to their respective
            db's.
    """

    def __init__(
        self,
        instances_db_file_path: Path,
        train_db_file_path: Path,
        valid_db_file_path: Optional[Path] = None,
        loader_batch_size: int = 48,
        loader_num_workers: int = 24,
        write_db_batch_size: int = 300000,
        example_id_prefix: str = "pretrain_",
        enabled_tasks: Optional[dict[MediaType, set[Task]]] = None,
    ) -> None:
        self._instance_counter: Counter[int] = Counter()

        self._example_id_prefix = example_id_prefix

        self._enabled_tasks = (
            enabled_tasks
            if enabled_tasks is not None
            else EnabledTasksHandler.get_default_enabled_tasks_per_modality()
        )

        self._dataset = IterableDatasetDbReader(
            instances_db_file_path, enabled_tasks=self._enabled_tasks
        )
        self._train_db = DatasetDb(
            train_db_file_path, readonly=False, batch_size=write_db_batch_size
        )

        self._valid_db = (
            DatasetDb(valid_db_file_path, readonly=False, batch_size=write_db_batch_size)
            if valid_db_file_path is not None
            else None
        )

        self._loader = DataLoader(
            self._dataset,
            collate_fn=_loader_collate_fn,
            batch_size=loader_batch_size,
            shuffle=False,
            num_workers=loader_num_workers,
        )

        self._progress = get_progress()
        self._get_instance_task_id = self._progress.add_task(
            "Instances processed", total=self._dataset.end
        )
        self._save_train_instance_task_id = self._progress.add_task(
            "Saving train instances", total=float("inf"), start=False
        )
        self._save_valid_instance_task_id = self._progress.add_task(
            "Saving valid instances",
            total=float("inf"),
            start=False,
            visible=self._valid_db is not None,
        )

    def run(self) -> None:
        """Run the preparation process."""
        with self._progress, self._train_db:  # noqa: WPS316
            self._progress.start_task(self._save_train_instance_task_id)

            if self._valid_db is None:
                self.process_train_instances()

            elif self._valid_db is not None:
                self._progress.start_task(self._save_valid_instance_task_id)

                with self._valid_db:
                    self.process_train_valid_instances()

    def process_train_instances(self) -> None:
        """Process all the training instances."""
        for batch in self._loader:
            for source_idx, pretrain_instance, is_train in batch:
                self.add_index_to_counter(source_idx)

                if is_train:
                    self.add_instance_to_train_db(pretrain_instance)

    def process_train_valid_instances(self) -> None:
        """Process both training and validation instances."""
        if self._valid_db is None:
            raise AssertionError(
                "`self.valid_db` is None, meaning it's not been initialised. Check the params used when setting up the class."
            )

        for batch in self._loader:
            for source_idx, pretrain_instance, is_train in batch:
                self.add_index_to_counter(source_idx)

                if is_train:
                    self.add_instance_to_train_db(pretrain_instance)
                else:
                    self.add_instance_to_valid_db(pretrain_instance)

    def add_instance_to_train_db(self, pretrain_instance: bytes) -> None:
        """Add the pretrain instance to the training db.

        The pretrain instance has already been compressed into bytes for the DatasetDb.
        """
        data_idx = self._progress.tasks[self._save_train_instance_task_id].completed
        key = (data_idx, f"{self._example_id_prefix}{data_idx}")

        self._train_db[key] = pretrain_instance

        self._progress.advance(self._save_train_instance_task_id)

    def add_instance_to_valid_db(self, pretrain_instance: bytes) -> None:
        """Add the pretrain instance to the validation db.

        The pretrain instance has already been compressed into bytes for the DatasetDb.
        """
        data_idx = self._progress.tasks[self._save_valid_instance_task_id].completed
        key = (data_idx, f"{self._example_id_prefix}{data_idx}")

        self._valid_db[key] = pretrain_instance

        self._progress.advance(self._save_valid_instance_task_id)

    def add_index_to_counter(self, source_idx: int) -> None:
        """Add the index of the source instance to the counter and update the progress bar."""
        should_update_progress_bar = source_idx not in self._instance_counter

        self._instance_counter[source_idx] += 1

        if should_update_progress_bar:
            self._progress.update(
                self._get_instance_task_id, completed=len(self._instance_counter.keys())
            )
