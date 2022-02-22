import json

from emma_datasets.common import Settings
from emma_datasets.common.logger import get_logger, use_rich_for_logging, use_rich_for_tracebacks

from emma_policy.datamodules.pretrain_datamodule import DEFAULT_COCO_SPLITS_PATH


MAX_INSTANCES = 100
SPLIT_FIXTURES_DIR = Settings().paths.storage.joinpath("fixtures", "vl-t5-splits")

logger = get_logger()
use_rich_for_logging()
use_rich_for_tracebacks()


def build_fixtures() -> None:
    """Generates a smaller files containing the VL-T5 pretraining splits."""
    SPLIT_FIXTURES_DIR.mkdir(exist_ok=True)

    for split_file in DEFAULT_COCO_SPLITS_PATH.glob("*.json"):
        logger.info(f"Generating fixtures for file: {split_file}")
        with open(split_file) as in_file:
            data_list = json.load(in_file)

            # from this dataset list we extract only the image ids
            output_file = SPLIT_FIXTURES_DIR.joinpath(split_file.name)

            with open(output_file, "w") as out_file:
                json.dump(data_list[:MAX_INSTANCES], out_file)


if __name__ == "__main__":
    build_fixtures()
