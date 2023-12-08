from pathlib import Path

from pydantic import AnyHttpUrl, BaseSettings


BASE_DIR = Path("storage/")


class Paths:
    """Dataclass for data paths."""

    def __init__(self, base_dir: Path = BASE_DIR) -> None:
        self.storage = base_dir

        self.databases = self.storage.joinpath("db/")
        self.model_checkpoints = self.storage.joinpath("model/")
        self.constants = self.storage.joinpath("constants/")

    def create_dirs(self) -> None:
        """Create directories for files if they do not exist."""
        self.model_checkpoints.mkdir(parents=True, exist_ok=True)
        self.databases.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Common settings class for use throughout the repository."""

    paths: Paths = Paths()
    feature_extractor_endpoint: AnyHttpUrl = "http://0.0.0.0:5500"  # type: ignore[assignment]

    class Config:
        """Inner config for API Settings to allow setting inner models from env variables."""

        env_nested_delimiter = "__"
