from pathlib import Path
from typing import Any

from pytest_cases import fixture


@fixture(scope="session")
def instances_tiny_batch_db_path(fixtures_root: Path) -> Path:
    return fixtures_root.joinpath("instances_tiny_batch.db")


@fixture(scope="session")
def cached_db_dir_path(request: Any) -> Path:
    """Cached dir path for storing DatasetDb files that are made once."""
    return Path(request.config.cache.makedir("db"))
