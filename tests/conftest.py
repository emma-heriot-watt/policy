import os
from pathlib import Path

import pytest
from emma_datasets.db import DatasetDb
from pytest_cases import fixture


if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@fixture
def tiny_instances_db_path(tmp_path: Path) -> Path:
    """Create an DatasetDb of instances with very few instances."""
    max_num_instances = 1

    tiny_instances_db_path = tmp_path.joinpath("tiny_instances.db")

    tiny_instances_db = DatasetDb(tiny_instances_db_path, readonly=False)

    with tiny_instances_db:
        with DatasetDb("storage/fixtures/instances.db") as instances_db:

            for data_id, example_id, instance_str in instances_db:
                if data_id > max_num_instances:
                    break

                tiny_instances_db[(data_id, example_id)] = instance_str

    return tiny_instances_db_path
