from pathlib import Path
from typing import Any

import pytest
from pytest_cases import fixture, param_fixture


@fixture(scope="session")
def project_root() -> Path:
    return Path.joinpath(Path(__file__).parent.parent, "..").resolve()


@fixture(scope="session")
def fixtures_root(project_root: Path) -> Path:
    return Path.joinpath(project_root, "storage", "fixtures")


@fixture(scope="session")
def cached_db_dir_path(request: Any) -> Path:
    """Create a cache for any DatasetDb's so they don't need to keep being re-built."""
    return Path(request.config.cache.makedir("db"))


@fixture(scope="session")
def instances_db_path(fixtures_root: Path) -> Path:
    return fixtures_root.joinpath("instances.db")


model_metadata_path = param_fixture(
    "model_metadata_path",
    [
        pytest.param("heriot-watt/emma-small", marks=pytest.mark.order(1), id="emma-small"),
        pytest.param("heriot-watt/emma-base", marks=pytest.mark.slow, id="emma-base"),
    ],
    scope="session",
)
