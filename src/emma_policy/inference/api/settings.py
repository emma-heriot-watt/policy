from argparse import ArgumentParser, Namespace
from pathlib import Path

from pydantic import BaseModel, BaseSettings


class FeatureExtractorSettings(BaseModel):
    """API settings for the Feature Extractor."""

    host: str = "http://0.0.0.0"
    port: int = 5500
    single_feature: str = "features"
    batch_features: str = "batch_features"

    @property
    def endpoint(self) -> str:
        """Get the endpoint of the API."""
        return f"{self.host}:{self.port}"

    def get_single_feature_url(self) -> str:
        """Get the URL to extract features from a single image."""
        return f"{self.endpoint}/{self.single_feature}"

    def get_batch_features_url(self) -> str:
        """Get the URL to extract features from a batch of images."""
        return f"{self.endpoint}/{self.batch_features}"


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 5000
    host: str = "0.0.0.0"  # noqa: S104
    log_level: str = "info"
    feature_extractor_api: FeatureExtractorSettings = FeatureExtractorSettings()

    class Config:
        """Inner config for API Settings to allow setting inner models from env variables."""

        env_nested_delimiter = "__"


def parse_api_args() -> tuple[Namespace, list[str]]:
    """Parse any arguments, with any extras being provided as model arguments."""
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help='Base data directory containing subfolders "games" and "edh_instances"',
    )
    arg_parser.add_argument(
        "--images_dir",
        type=Path,
        required=True,
        help="Images directory containing inference image output",
    )
    arg_parser.add_argument(
        "--split",
        type=str,
        default="valid_seen",
        choices=["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"],
        help="One of train, valid_seen, valid_unseen, test_seen, test_unseen",
    )

    return arg_parser.parse_known_args()
