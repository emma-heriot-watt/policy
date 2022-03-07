from pathlib import Path

from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb
from pytest_cases import fixture

from emma_policy.datamodules.pretrain_instances.convert_to_pretrain_instances import (
    convert_instance_to_pretrain_instances,
)
from emma_policy.datamodules.pretrain_instances.datamodels import PretrainInstance


@fixture(scope="session")
def instances_tiny_batch_db_path(fixtures_root: Path) -> Path:
    return fixtures_root.joinpath("instances_tiny_batch.db")


@fixture
def tiny_instances_db_path(tmp_path: Path) -> Path:
    """Create an DatasetDb of instances with very few instances."""
    max_num_instances = 2

    tiny_instances_db_path = tmp_path.joinpath("tiny_instances.db")

    tiny_instances_db = DatasetDb(tiny_instances_db_path, readonly=False)

    with tiny_instances_db:
        with DatasetDb("storage/fixtures/instances.db") as instances_db:

            for data_id, example_id, instance_str in instances_db:
                if data_id > max_num_instances:
                    break

                tiny_instances_db[(data_id, example_id)] = instance_str

    return tiny_instances_db_path


@fixture
def tiny_pretrain_db_path(tmp_path: Path) -> Path:
    """Create an DatasetDb of instances with very few instances."""
    tiny_pretrain_db_path = tmp_path.joinpath("tiny_pretrain.db")

    tiny_pretrain_db = DatasetDb(tiny_pretrain_db_path, readonly=False)

    with tiny_pretrain_db:

        max_num_instances = 10

        with DatasetDb("storage/fixtures/instances.db") as db:
            data_id = 0
            for _, _, instance_str in db:
                instance = Instance.parse_raw(instance_str)

                pretrain_instances = convert_instance_to_pretrain_instances(instance)

                for pretrain_instance in pretrain_instances:
                    assert isinstance(pretrain_instance, PretrainInstance)

                    if data_id > max_num_instances:
                        break  # noqa: WPS220

                    example_id = f"pretrain_{data_id}"
                    tiny_pretrain_db[(data_id, example_id)] = pretrain_instance
                    data_id += 1

    return tiny_pretrain_db_path
