from pathlib import Path

from emma_datasets.common import get_progress
from emma_datasets.db import DatasetDb
from emma_datasets.parsers.instance_creators import TeachEdhInstanceCreator
from pytest_cases import fixture


@fixture(scope="session")
def instances_tiny_batch_db_path(fixtures_root: Path) -> Path:
    return fixtures_root.joinpath("instances_tiny_batch.db")


@fixture(scope="session")
def teach_edh_instances_db(cached_db_dir_path: Path, fixtures_root: Path) -> Path:
    """Create an DatasetDb of TEACh EDH instances and cache to use across tests."""
    progress = get_progress()

    instance_creator = TeachEdhInstanceCreator(progress)
    instance_iterator = instance_creator(
        input_data=fixtures_root.joinpath("teach_edh").rglob("*.json"),
        progress=progress,
    )

    teach_instances_db_path = cached_db_dir_path.joinpath("teach_instances.db")
    db = DatasetDb(teach_instances_db_path, readonly=False)

    with db:
        for idx, instance in enumerate(instance_iterator):
            db[(idx, f"teach_edh_{idx}")] = instance

    return teach_instances_db_path
