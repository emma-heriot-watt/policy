from pathlib import Path
from typing import Any
from unittest.mock import PropertyMock

from emma_datasets.common import get_progress
from emma_datasets.datamodels.datasets.teach import TeachEdhInstance
from emma_datasets.db import DatasetDb
from emma_datasets.parsers.instance_creators import TeachEdhInstanceCreator
from pytest_cases import fixture
from pytest_mock import MockerFixture


@fixture(scope="session")
def instances_tiny_batch_db_path(fixtures_root: Path) -> Path:
    return fixtures_root.joinpath("instances_tiny_batch.db")


class TeachEdhInstanceFeaturesPathPropertyMock(PropertyMock):  # type: ignore[misc]
    """Mock the `features_path` property within the TeachEdhInstance.

    The features path within each instance is derived automatically and NOT hard-coded into the
    class. Therefore to be able to test the instances properly, we need to override the property
    and return something else.
    """

    def __get__(self, obj: TeachEdhInstance, obj_type: Any = None) -> Path:  # noqa: WPS110
        """Get the features path from the fixtures.

        This updates the `return_value`, which is used by `unittest.Mock` to return a value.
        """
        dataset_index = obj._features_path.parts.index("datasets")
        self.return_value = Path("storage", "fixtures", *obj._features_path.parts[dataset_index:])
        return self()


@fixture(scope="session")
def teach_edh_instances_db(
    cached_db_dir_path: Path, fixtures_root: Path, session_mocker: MockerFixture
) -> Path:
    """Create an DatasetDb of TEACh EDH instances and cache to use across tests.

    Additionally, this fixture also mocks the features path of each TeachEdhInstance to point to
    the fixtures dir.
    """
    session_mocker.patch.object(
        TeachEdhInstance,
        "features_path",
        new_callable=TeachEdhInstanceFeaturesPathPropertyMock,
    )

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
