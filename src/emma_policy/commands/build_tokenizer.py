import itertools
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterator

from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb
from transformers import AutoTokenizer

from emma_policy.common import get_logger
from emma_policy.datamodules.pretrain_instances.datamodels import (
    TASK_TEMPLATES_MAP,
    extract_task_prefix_strings,
)


logger = get_logger(__name__)


def create_data_iterator(db_path: Path) -> Iterator[str]:
    """Opens the dataset and create an iterator over all the language annotations."""
    with DatasetDb(db_path) as db:
        for _, _, raw_data in db:
            data = Instance.parse_raw(raw_data)

            yield from data.language_annotations


def main(args: Namespace) -> None:
    """Trains a new tokenizer."""
    logger.info(f"Loading data from dataset {args.db_path}")
    data_iterator = create_data_iterator(args.db_path)

    if args.num_test_instances is not None:
        data_iterator = itertools.islice(data_iterator, 0, args.num_test_instances)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f"Created {type(tokenizer)} tokenizer")

    object_tokens = [f"<vis_token_{idx}>" for idx in range(1, args.num_visual_tokens + 1)]

    tokenizer = tokenizer.train_new_from_iterator(
        itertools.chain(extract_task_prefix_strings(TASK_TEMPLATES_MAP), data_iterator),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        new_special_tokens=object_tokens,
    )

    logger.info(f"Saving tokenizer to path {args.output_path}")
    args.output_path.mkdir(exist_ok=True, parents=True)
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--db_path", type=Path, help="Path to a DatasetDb file", default="storage/db/instances.db"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="allenai/led-base-16384",
        choices=("allenai/led-base-16384", "facebook/bart-base"),
        help="The type of tokenizer to train",
    )
    parser.add_argument(
        "--num_visual_tokens",
        type=int,
        default=100,
        help="Number of total visual tokens for each visual frame.",
    )
    parser.add_argument("--vocab_size", type=int, default=10000)  # noqa: WPS432
    parser.add_argument("--min_frequency", type=int, default=0)
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Tokenizer output path",
        default="storage/tokenizer/emma",
    )
    parser.add_argument(
        "--num_test_instances",
        type=int,
        default=None,
        help="Only returns this number of test instances from the data iterator",
    )

    args = parser.parse_args()
    main(args)
