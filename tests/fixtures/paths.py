from pathlib import Path

import pytest
from pytest_cases import fixture, param_fixture


@fixture(scope="session")
def project_root() -> Path:
    return Path.joinpath(Path(__file__).parent.parent, "..").resolve()


@fixture(scope="session")
def fixtures_root(project_root: Path) -> Path:
    return Path.joinpath(project_root, "storage", "fixtures")


model_metadata_path = param_fixture(
    "model_metadata_path",
    [
        pytest.param("heriot-watt/emma-small", id="emma-small"),
        pytest.param("heriot-watt/emma-base", marks=pytest.mark.slow, id="emma-base"),
    ],
    scope="session",
)
