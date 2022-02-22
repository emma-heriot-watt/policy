from argparse import Namespace
from pathlib import Path

import pytest
from pytest_cases import fixture_ref, parametrize

from emma_policy.commands.build_tokenizer import main as build_tokenizer_main


@parametrize(
    "instances_db_path",
    [
        pytest.param(Path("storage/fixtures/instances.db"), marks=pytest.mark.slow, id="full"),
        pytest.param(fixture_ref("tiny_instances_db_path"), id="subset"),
    ],
)
@parametrize("tokenizer", ["allenai/led-base-16384", "facebook/bart-base"])
@parametrize("num_visual_tokens", [100])
@parametrize("num_test_instances", [100])
@parametrize("vocab_size", [10000])
@parametrize("min_frequency", [0])
def test_build_tokenizer(  # noqa: WPS216
    tmp_path: Path,
    instances_db_path: Path,
    tokenizer: str,
    num_visual_tokens: int,
    num_test_instances: int,
    vocab_size: int,
    min_frequency: int,
) -> None:
    args = Namespace(
        output_path=tmp_path.joinpath("tokenizer"),
        db_path=instances_db_path,
        tokenizer=tokenizer,
        num_test_instances=num_test_instances,
        num_visual_tokens=num_visual_tokens,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    build_tokenizer_main(args)
