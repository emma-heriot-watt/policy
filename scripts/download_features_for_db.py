import argparse
import logging
from pathlib import Path

import boto3
import botocore
from emma_datasets.common import get_progress
from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb

from emma_policy.utils import get_logger


log = get_logger(__name__)

logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("nose").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def main(args: argparse.Namespace) -> None:
    """Downloads the features for all instances given a db."""
    s3 = boto3.client("s3")

    progress = get_progress()

    with progress:
        with DatasetDb(args.input_db) as in_db:
            task_id = progress.add_task(
                f"Downloading features from {args.input_db}", total=len(in_db)
            )
            for _, _, data in in_db:
                instance = Instance.parse_raw(data)
                local_feature_path = instance.features_path
                if "alfred" in local_feature_path.parts:
                    # Will probably need something like that for teach as well
                    idx2split = local_feature_path.parts.index("alfred")
                    feature_path = Path(*local_feature_path.parts[idx2split + 1 :])
                    s3_parts = ("datasets", "alfred", "full_2.1.0", feature_path)
                    s3_feature_path = Path(*s3_parts)
                else:
                    idx2split = local_feature_path.parts.index("datasets")
                    feature_path = Path(*local_feature_path.parts[idx2split + 1 :])
                    s3_feature_path = Path.joinpath(Path("datasets"), feature_path)

                Path.mkdir(local_feature_path.parent, parents=True, exist_ok=True)
                try:
                    s3.download_file("emma-simbot", str(s3_feature_path), str(local_feature_path))
                except botocore.exceptions.ClientError:
                    log.error(f"Failed to download {local_feature_path}")
                progress.advance(task_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_db",
        help="Path to the input database",
        type=Path,
        default="storage/fixtures/instances.db",
    )

    args = parser.parse_args()
    main(args)
