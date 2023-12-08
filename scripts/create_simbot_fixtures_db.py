import json
import random
from argparse import ArgumentParser
from pathlib import Path

from emma_datasets.common.progress import BatchesProcessedColumn, ProcessingSpeedColumn
from emma_datasets.datamodels.datasets.simbot import SimBotInstructionInstance
from emma_datasets.db import DatasetDb
from rich.progress import BarColumn, Progress

from emma_policy.datamodules.pretrain_instances import PretrainInstanceCreator, Task


class FixturesDbCreator:
    """Create the instances db for the fixtures accounting for all the tasks."""

    def __init__(
        self,
        input_db_path: Path,
        output_db_path: Path,
        fixtures_datasets_path: Path,
        min_train_instances: int,
        min_valid_instances: int,
    ) -> None:
        self.in_db = DatasetDb(input_db_path)
        self.out_db = DatasetDb(output_db_path, readonly=False)

        self._fixtures_datasets_path = fixtures_datasets_path

        self._new_data_idx = 0

        self.progress = Progress(
            "{task.description}",
            BarColumn(),
            BatchesProcessedColumn(),
            ProcessingSpeedColumn(),
        )
        self._overall_task = self.progress.add_task("Iterating instances", total=len(self.in_db))

        task = Task.action_execution

        self._tasks = {
            task: self.progress.add_task(f"[b]{task.value}[/] (train)", total=min_train_instances)
        }

        self._index_order = list(range(len(self.in_db)))
        random.shuffle(self._index_order)

    @property
    def is_done(self) -> bool:
        """Check whether the output db has all the required instances."""
        all_task_ids = list(self._tasks.values())
        return all([self.progress.tasks[task].finished for task in all_task_ids])

    def run(self) -> None:
        """Process all the instances."""
        with self.progress, self.out_db:  # noqa: WPS316
            self.process()

    def process(self) -> None:
        """Process all the instances."""
        for index in self._index_order:
            self.progress.advance(self._overall_task)
            instance_str = self.in_db[index]

            instance = SimBotInstructionInstance.parse_raw(instance_str)
            instance = self._fix_features_path_for_fixtures(instance)

            self.process_single_instance(instance)

            if self.is_done:
                break

    def process_single_instance(self, instance: SimBotInstructionInstance) -> None:
        """Process a single instance and its task."""
        tasks = [Task.action_execution]

        is_train_tasks_done = self._is_all_tasks_finished(tasks)

        if not is_train_tasks_done:
            for task in tasks:
                self.progress.advance(self._tasks[task])

            return self._add_instance_to_db(instance)

    def _add_instance_to_db(self, instance: SimBotInstructionInstance) -> None:
        """Add the instance to the output db."""
        self.out_db[(self._new_data_idx, f"pretrain_{self._new_data_idx}")] = instance
        self._new_data_idx += 1

    def _get_all_tasks_from_instance(self, instance: SimBotInstructionInstance) -> list[Task]:
        """Get all tasks from the instance.

        This uses the same logic for converting the pretraining instances, and returns a list of
        all the valid tasks.
        """
        creator = PretrainInstanceCreator(instance, None)

        pretrain_instance_list_per_task = {
            task: list(instance_iterator)
            for task, instance_iterator in creator.instance_task_map.items()
        }

        tasks_for_instance = [
            task
            for task, pretrain_instance_list in pretrain_instance_list_per_task.items()
            if pretrain_instance_list
        ]

        return tasks_for_instance

    def _fix_features_path_for_fixtures(
        self, instance: SimBotInstructionInstance
    ) -> SimBotInstructionInstance:
        """Replace the features path for each dataset metadata to point to the fixtures.

        Workaround on immutable objects, change the feature path from the string rather than the
        instance value. Does not affect the path to the image files.
        """
        dirname = self._get_features_dirname(instance)
        instance_dict = json.loads(instance.json(by_alias=True))
        instance_dict["features_path"] = self._update_features_path(dirname=dirname)

        instance_dict = json.dumps(instance_dict)
        instance_dict = instance_dict.replace("None", "null")
        return SimBotInstructionInstance.parse_raw(instance_dict)

    def _get_features_dirname(self, instance: SimBotInstructionInstance) -> Path:
        """Get the name of the features file."""
        dirname = Path(*instance.features_path.parts[2:])

        return dirname

    def _update_features_path(self, dirname: Path) -> str:
        """Update the features path for the fixtures."""
        features_path = self._fixtures_datasets_path.joinpath(dirname).as_posix()

        return features_path

    def _is_all_tasks_finished(self, tasks: list[Task]) -> bool:
        """Check whether all the given tasks have finished.

        If at least one is not, then it returns False and the instance is saved. This is because we
        do not care about having a balanced set of tasks, but rather having instances that are
        suitable for that task to hope tests are better.
        """
        is_task_finished = (self.progress.tasks[self._tasks[task]].finished for task in tasks)

        return all(is_task_finished)


def main(
    input_db_path: Path,
    output_db_path: Path,
    output_features_path: Path,
    min_train_instances: int = 100,
    min_valid_instances: int = 20,
) -> None:
    """Create the fixtures db."""
    creator = FixturesDbCreator(
        input_db_path,
        output_db_path,
        output_features_path,
        min_train_instances,
        min_valid_instances,
    )
    creator.run()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--input_db",
        type=Path,
        help="Path to the input database",
        default="storage/db/simbot_instructions_train.db",
    )
    parser.add_argument(
        "--output_db",
        type=Path,
        help="Path to the output database",
        default="storage/fixtures/db/simbot_instructions_train.db",
    )

    parser.add_argument(
        "--min_train_instances",
        type=int,
        help="Minimum number of train instances",
        default=10,
    )

    parser.add_argument(
        "--min_valid_instances",
        type=int,
        help="Minimum number of valid instances",
        default=5,
    )

    parser.add_argument(
        "--output_features_path",
        type=Path,
        help="Path to the output database",
        default="storage/fixtures/datasets/",
    )

    args = parser.parse_args()
    main(
        input_db_path=args.input_db,
        output_db_path=args.output_db,
        output_features_path=args.output_features_path,
        min_train_instances=args.min_train_instances,
        min_valid_instances=args.min_valid_instances,
    )
