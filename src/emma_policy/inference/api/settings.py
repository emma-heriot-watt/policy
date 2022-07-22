from argparse import ArgumentParser, Namespace
from pathlib import Path

from pydantic import AnyHttpUrl, BaseSettings


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 5000
    host: str = "0.0.0.0"  # noqa: S104
    log_level: str = "info"
    feature_extractor_endpoint: AnyHttpUrl = "http://0.0.0.0:5500"  # type: ignore[assignment]


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
