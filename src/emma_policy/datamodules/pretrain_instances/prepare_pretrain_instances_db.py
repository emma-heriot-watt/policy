import math
from collections import Counter
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import torch
from emma_datasets.datamodels import DatasetSplit, Instance, MediaType
from emma_datasets.db import DatasetDb, JsonStorage
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import SpinnerColumn, TaskID
from torch.utils.data import DataLoader, IterableDataset

from emma_policy.common import (
    BatchesProcessedColumn,
    CustomBarColumn,
    CustomProgress,
    CustomTimeColumn,
    ProcessingSpeedColumn,
)
from emma_policy.datamodules.pretrain_instances import convert_instance_to_pretrain_instances
from emma_policy.datamodules.pretrain_instances.datamodels import EnabledTasksHandler, Task
from emma_policy.datamodules.pretrain_instances.is_train_instance import is_train_instance


PRETRAIN_DATASET_SPLITS = (DatasetSplit.train, DatasetSplit.valid)


def get_db_file_name(task: Task, dataset_split: DatasetSplit) -> str:
    """Get the name of the DB file for a given task and dataset split."""
    return f"{task.name}_{dataset_split.name}.db"


class DatasetDbReaderReturn(NamedTuple):
    """Return tuple for the DatasetDbReader."""

    source_instance_idx: int
    pretrain_instance: bytes
    is_train: bool
    task: Task


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
                    task=pretrain_instance.task,
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
        output_dir_path (Path): Output directory to save all DBs.
        loader_num_workers (int): Number of workers for the torch DataLoader. Defaults to 0.
        loader_batch_size_per_worker (int): Batch size for each worker in the torch DataLoader.
            This ensures that all workers are used to their maximum and none are left waiting for
            another to finish processing. Defaults to 5.
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
        output_dir_path: Path,
        loader_num_workers: int = 0,
        loader_batch_size_per_worker: int = 5,
        write_db_batch_size: int = 300000,
        example_id_prefix: str = "pretrain_",
    ) -> None:
        self._dataset = IterableDatasetDbReader(
            instances_db_file_path,
            enabled_tasks=EnabledTasksHandler.get_default_enabled_tasks_per_modality(),
        )

        self._loader = DataLoader(
            self._dataset,
            collate_fn=_loader_collate_fn,
            batch_size=max(
                loader_batch_size_per_worker,
                loader_num_workers * loader_batch_size_per_worker,
            ),
            shuffle=False,
            num_workers=loader_num_workers,
        )

        self._example_id_prefix = example_id_prefix

        self._instance_counter: Counter[int] = Counter()

        self._overall_progress = CustomProgress(
            "[progress.description]{task.description}",
            CustomBarColumn(),
            BatchesProcessedColumn(),
            CustomTimeColumn(),
            ProcessingSpeedColumn(),
        )

        self._task_progress = CustomProgress(
            "[progress.description]{task.description}",
            SpinnerColumn(),
            BatchesProcessedColumn(),
        )

        self._output_dbs: dict[Task, dict[DatasetSplit, DatasetDb]] = {
            task: {
                split: DatasetDb(
                    output_dir_path.joinpath(get_db_file_name(task, split)),
                    readonly=False,
                    batch_size=write_db_batch_size,
                )
                for split in PRETRAIN_DATASET_SPLITS
            }
            for task in Task
        }

        self._overall_task_id = self._overall_progress.add_task(
            "Instances processed", total=self._dataset.end, start=False
        )

        self._task_ids: dict[Task, dict[DatasetSplit, TaskID]] = {
            task: {
                split: self._task_progress.add_task(
                    f"{task.value} ({split.name})",
                    total=float("inf"),
                    start=False,
                )
                for split in PRETRAIN_DATASET_SPLITS
            }
            for task in Task
        }

    def run(self) -> None:
        """Run the preparation process and save all the DBs."""
        with self._display_progress():
            self._start_tasks()
            self.process_instances()

        self._close_all_dbs()

    def process_instances(self) -> None:
        """Process all the instances and make sure to save to the correct DB."""
        for batch in self._loader:
            for source_idx, pretrain_instance, is_train, task in batch:
                self.add_index_to_counter(source_idx)

                if is_train:
                    self.add_instance_to_db(pretrain_instance, task, DatasetSplit.train)
                else:
                    self.add_instance_to_db(pretrain_instance, task, DatasetSplit.valid)

    def add_instance_to_db(
        self, pretrain_instance: bytes, task: Task, dataset_split: DatasetSplit
    ) -> None:
        """Append the instance to the DB for the given task/split.

        The progress bar is used directly to get the next data index for the DB.
        """
        progress_task_id = self._task_ids[task][dataset_split]

        # Get the data_idx from the progress bar and create the key
        data_idx = int(self._task_progress.tasks[progress_task_id].completed)
        key = (data_idx, f"{self._example_id_prefix}{data_idx}")

        self._output_dbs[task][dataset_split][key] = pretrain_instance

        self._task_progress.advance(progress_task_id)

    def add_index_to_counter(self, source_idx: int) -> None:
        """Add the index of the source instance to the counter and update the progress bar."""
        should_update_progress_bar = source_idx not in self._instance_counter

        self._instance_counter[source_idx] += 1

        if should_update_progress_bar:
            self._overall_progress.update(
                self._overall_task_id, completed=len(self._instance_counter.keys())
            )

    def _start_tasks(self) -> None:
        """Start all the progress tasks."""
        self._overall_progress.start_task(self._overall_task_id)

        for task_id_per_split in self._task_ids.values():
            for task_id in task_id_per_split.values():
                self._task_progress.start_task(task_id)

    def _close_all_dbs(self) -> None:
        """Close all the DBs.

        This makes sure that everything has been saved and closed properly.
        """
        for task_dbs in self._output_dbs.values():
            for db in task_dbs.values():
                db.close()

    def _display_progress(self) -> Live:
        """Return a rich `Live` object to display the progress bars.

        This should be used as a context manager.
        """
        progress_group = Group(
            Panel(self._task_progress),
            self._overall_progress,
        )

        return Live(progress_group)
