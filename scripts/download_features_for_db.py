import argparse
import logging
from pathlib import Path
from typing import Literal

import boto3
import botocore
from emma_datasets.common import get_progress
from emma_datasets.datamodels import BaseInstance, Instance, TeachEdhInstance
from emma_datasets.db import DatasetDb

from emma_policy.utils import get_logger


log = get_logger(__name__)

logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("nose").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


class FixtureDownload:
    """Downloads the features for all instances given a db."""

    def __init__(
        self, input_db_path: Path, instance_type: Literal["default", "teach_edh"] = "default"
    ) -> None:
        self._db = DatasetDb(input_db_path)

        self._instance_model: BaseInstance = Instance
        self.instance_type = instance_type
        if self.instance_type == "teach_edh":
            self._instance_model = TeachEdhInstance

        self._s3 = boto3.client("s3")

        self._progress = get_progress()
        self._task_id = self._progress.add_task(
            f"Downloading features from {input_db_path}", total=len(self._db)
        )

    def run(self) -> None:
        """Do the downloading for the fixtures."""
        with self._progress, self._db:  # noqa: WPS316
            for _, _, data in self._db:
                instance = self._instance_model.parse_raw(data)
                local_feature_path = instance.features_path

                local_feature_path.parent.mkdir(parents=True, exist_ok=True)

                if "alfred" in local_feature_path.parts:
                    s3_path = self._get_paths_for_alfred(local_feature_path)
                else:
                    s3_path = self._get_paths(local_feature_path)

                self._download_file(s3_path, local_feature_path)

                if self.instance_type == "teach_edh":
                    local_feature_path = instance.future_features_path
                    local_feature_path.parent.mkdir(parents=True, exist_ok=True)
                    s3_path = self._get_paths(local_feature_path)
                    self._download_file(s3_path, local_feature_path)

                self._progress.advance(self._task_id)

    def _download_file(self, s3_path: Path, local_path: Path) -> None:
        try:
            self._s3.download_file("emma-simbot", str(s3_path), str(local_path))
        except botocore.exceptions.ClientError:
            log.error(f"Failed to download {local_path}")

    def _get_paths(self, local_feature_path: Path) -> Path:
        """Get the paths as is, without needing any special handling."""
        idx2split = local_feature_path.parts.index("datasets")
        feature_path = Path(*local_feature_path.parts[idx2split + 1 :])
        s3_feature_path = Path("datasets", feature_path)

        return s3_feature_path

    def _get_paths_for_alfred(self, local_feature_path: Path) -> Path:
        idx2split = local_feature_path.parts.index("alfred")
        feature_path = Path(*local_feature_path.parts[idx2split + 1 :])
        s3_feature_path = Path("datasets", "alfred", "full_2.1.0", feature_path)

        return s3_feature_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_db",
        help="Path to the input database",
        type=Path,
        default="storage/fixtures/instances.db",
    )

    parser.add_argument(
        "--instance_type",
        help="Type of instance model used within the DatasetDb",
        choices=["default", "teach_edh"],
        default="default",
    )

    args = parser.parse_args()

    downloader = FixtureDownload(args.input_db, args.instance_type)
    downloader.run()
