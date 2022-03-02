from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Optional

from emma_datasets.common import get_progress
from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb

from emma_policy.datamodules.pretrain_instances import (
    DEFAULT_COCO_SPLITS_PATH,
    Task,
    is_train_instance,
    load_ref_coco_images,
)


def check_type(instance: Instance) -> Optional[Task]:
    """Checks the type of this instance.

    Only some types are handled for simplicity.
    """
    if instance.caption is not None:
        return Task.captioning

    if instance.qa is not None:
        return Task.vqa

    if instance.regions is not None:
        return Task.dense_captioning

    if instance.trajectory is not None:
        return Task.instruction_prediction

    return None


def main(args: Namespace, num_train_instances: int = 100, num_valid_instances: int = 20) -> None:
    """Create the fixtures pretraining database."""
    instance_counter: dict[Task, int] = defaultdict(int)
    valid_instance_counter: dict[Task, int] = defaultdict(int)
    new_data_idx = 0

    train_valid_splits = load_ref_coco_images(DEFAULT_COCO_SPLITS_PATH)

    progress = get_progress()
    in_db = DatasetDb(args.input_db)
    out_db = DatasetDb(args.output_db, readonly=False)

    task_id = progress.add_task("Creating test instance database", total=len(in_db))

    with in_db, out_db, progress:  # noqa: WPS316
        for _, _, data in in_db:
            instance = Instance.parse_raw(data)
            instance_type = check_type(instance)

            if instance_type is not None:
                instance_counter[instance_type] += 1

                if instance_counter[instance_type] < num_train_instances:
                    out_db[(new_data_idx, f"pretrain_{new_data_idx}")] = instance
                    new_data_idx += 1
                elif (  # noqa: WPS337
                    not is_train_instance(train_valid_splits, instance)
                    and valid_instance_counter[instance_type] < num_valid_instances
                ):
                    valid_instance_counter[instance_type] += 1
                    out_db[(new_data_idx, f"pretrain_{new_data_idx}")] = instance
                    new_data_idx += 1

            progress.advance(task_id)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--input_db",
        type=str,
        help="Path to the input database",
        default="storage/db/instances.db",
    )
    parser.add_argument(
        "--output_db",
        type=str,
        help="Path to the output database",
        default="storage/fixtures/instances.db",
    )

    args = parser.parse_args()
    main(args)
