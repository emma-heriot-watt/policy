import random
from argparse import ArgumentParser
from pathlib import Path

from emma_datasets.common.progress import BatchesProcessedColumn, ProcessingSpeedColumn
from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb
from rich.progress import BarColumn, Progress

from emma_policy.datamodules.pretrain_instances import (
    DEFAULT_COCO_SPLITS_PATH,
    PretrainInstanceCreator,
    Task,
    is_train_instance,
    load_ref_coco_images,
)


class FixturesDbCreator:
    """Create the instances db for the fixtures accounting for all the tasks."""

    def __init__(
        self,
        input_db_path: Path,
        output_db_path: Path,
        max_train_instances: int,
        max_valid_instances: int,
    ) -> None:
        self.in_db = DatasetDb(input_db_path)
        self.out_db = DatasetDb(output_db_path, readonly=False)

        self._train_valid_splits = load_ref_coco_images(DEFAULT_COCO_SPLITS_PATH)
        self._new_data_idx = 0

        self.progress = Progress(
            "{task.description}",
            BarColumn(),
            BatchesProcessedColumn(),
            ProcessingSpeedColumn(),
        )
        self.overall_task = self.progress.add_task("Iterating instances", total=len(self.in_db))

        self.train_tasks = {
            task: self.progress.add_task(f"[b]{task.value}[/] (train)", total=max_train_instances)
            for task in Task
        }

        self.valid_tasks = {
            task: self.progress.add_task(f"[b]{task.value}[/] (valid)", total=max_valid_instances)
            for task in Task
            if task not in {Task.instruction_prediction, Task.action_execution, Task.vtm}
        }

        self._index_order = list(range(len(self.in_db)))
        random.shuffle(self._index_order)

    @property
    def is_done(self) -> bool:
        """Check whether the output db has all the required instances."""
        all_task_ids = list(self.train_tasks.values()) + list(self.valid_tasks.values())
        return all([self.progress.tasks[task].finished for task in all_task_ids])

    def run(self) -> None:
        """Process all the instances."""
        with self.progress, self.out_db:  # noqa: WPS316
            self.process()

    def process(self) -> None:
        """Process all the instances."""
        for index in self._index_order:
            self.progress.advance(self.overall_task)

            instance_str = self.in_db[index]
            instance = Instance.parse_raw(instance_str)
            instance_task = self._get_task_from_instance(instance)

            if instance_task is not None:
                if instance_task not in self.valid_tasks.keys():
                    self.process_single_instance_for_training_only(instance, instance_task)
                else:
                    self.process_single_instance(instance, instance_task)

            if self.is_done:
                break

    def process_single_instance_for_training_only(self, instance: Instance, task: Task) -> None:
        """Process a single instance for a task that does not have validation data."""
        is_train_task_done = self.progress.tasks[self.train_tasks[task]].finished

        if not is_train_task_done:
            self.progress.advance(self.train_tasks[task])
            return self._add_instance_to_db(instance)

    def process_single_instance(self, instance: Instance, task: Task) -> None:
        """Process a single instance and its task."""
        is_instance_in_train_set = is_train_instance(self._train_valid_splits, instance)

        is_train_task_done = self.progress.tasks[self.train_tasks[task]].finished
        is_valid_task_done = self.progress.tasks[self.valid_tasks[task]].finished

        if not is_train_task_done and is_instance_in_train_set:
            self.progress.advance(self.train_tasks[task])
            return self._add_instance_to_db(instance)

        if not is_valid_task_done and not is_instance_in_train_set:
            self.progress.advance(self.valid_tasks[task])
            return self._add_instance_to_db(instance)

    def _add_instance_to_db(self, instance: Instance) -> None:
        """Add the instance to the output db."""
        self.out_db[(self._new_data_idx, f"pretrain_{self._new_data_idx}")] = instance
        self._new_data_idx += 1

    def _get_task_from_instance(self, instance: Instance) -> Task:
        """Get the task from the instance.

        This uses the same logic for converting the pretraining instances, and returns a task that
        is valid.
        """
        creator = PretrainInstanceCreator(instance, None)

        pretrain_instance_list_per_task = {
            task: list(instance_iterator)
            for task, instance_iterator in creator.instance_task_map.items()
        }

        tasks_for_instance = [
            task
            for task, pretrain_instance_list in pretrain_instance_list_per_task.items()
            if not pretrain_instance_list
        ]

        return random.choice(tasks_for_instance)


def main(
    input_db_path: Path,
    output_db_path: Path,
    max_train_instances: int = 100,
    max_valid_instances: int = 20,
) -> None:
    """Create the fixtures db."""
    creator = FixturesDbCreator(
        input_db_path, output_db_path, max_train_instances, max_valid_instances
    )
    creator.run()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--input_db",
        type=Path,
        help="Path to the input database",
        default="storage/db/instances.db",
    )
    parser.add_argument(
        "--output_db",
        type=Path,
        help="Path to the output database",
        default="storage/fixtures/instances.db",
    )

    args = parser.parse_args()
    main(args.input_db, args.output_db)
