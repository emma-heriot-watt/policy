import itertools
from pathlib import Path
from typing import Any
from unittest.mock import PropertyMock

from emma_datasets.datamodels.datasets import TeachEdhInstance
from filelock import FileLock
from pytest_cases import fixture

from emma_policy.commands.create_pretrain_dbs import create_pretrain_dbs
from emma_policy.datamodules.pretrain_instances import (
    PRETRAIN_DATASET_SPLITS,
    Task,
    get_db_file_name,
)


@fixture(scope="session")
def pretrain_db_dir_path(cached_db_dir_path: Path, instances_db_path: Path) -> Path:
    """Create and cache the various task-specific pretrain DBs.

    This fixture will only create the DBs if the files do not already exist.
    """
    with FileLock(cached_db_dir_path.joinpath("pretrain_db.lock")):
        all_dbs_exist = all(
            cached_db_dir_path.joinpath(get_db_file_name(task, dataset_split)).exists()
            for task, dataset_split in itertools.product(Task, PRETRAIN_DATASET_SPLITS)
        )

        if not all_dbs_exist:
            create_pretrain_dbs(instances_db_path, cached_db_dir_path)

    return cached_db_dir_path


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


class TeachEdhInstanceFutureFeaturesPathPropertyMock(PropertyMock):  # type: ignore[misc]
    """Mock the `future_features_path` property within the TeachEdhInstance.

    The future features path within each instance is derived automatically and NOT hard-coded into
    the class. Therefore to be able to test the instances properly, we need to override the
    property and return something else.
    """

    def __get__(self, obj: TeachEdhInstance, obj_type: Any = None) -> Path:  # noqa: WPS110
        """Get the future features path from the fixtures.

        This updates the `return_value`, which is used by `unittest.Mock` to return a value.
        """
        dataset_index = obj._features_path.parts.index("datasets")
        self.return_value = Path(
            "storage",
            "fixtures",
            *obj._future_features_path.parts[dataset_index:],
        )
        return self()
