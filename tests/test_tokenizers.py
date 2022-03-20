from argparse import Namespace
from pathlib import Path

from pytest_cases import parametrize

from emma_policy.commands.build_tokenizer import main as build_tokenizer_main
from emma_policy.models.tokenizer_emma import EmmaTokenizer


@parametrize("tokenizer", ["allenai/led-base-16384", "facebook/bart-base"])
@parametrize("num_visual_tokens", [100])
@parametrize("num_frame_tokens", [250])
@parametrize("num_test_instances", [100])
@parametrize("vocab_size", [10000])
@parametrize("min_frequency", [0])
def test_build_tokenizer(  # noqa: WPS216
    tmp_path: Path,
    instances_db_path: Path,
    tokenizer: str,
    num_visual_tokens: int,
    num_frame_tokens: int,
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
        num_frame_tokens=num_frame_tokens,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    build_tokenizer_main(args)


def test_tokenize_input(emma_tokenizer: EmmaTokenizer) -> None:
    phrase = "The dog is eating an icecream."
    encodings = emma_tokenizer.encode_plus(phrase)

    tokens = emma_tokenizer.convert_ids_to_tokens(encodings.input_ids)
    conv_phrase = (
        emma_tokenizer.convert_tokens_to_string(tokens)
        .replace(emma_tokenizer.cls_token, "")
        .replace(emma_tokenizer.eos_token, "")
    )
    assert phrase == conv_phrase


@parametrize("max_length", [5])
def test_tokenizer_truncates_input(emma_tokenizer: EmmaTokenizer, max_length: int) -> None:
    phrase = "The dog is eating an icecream."
    emma_tokenizer.model_max_length = max_length
    emma_tokenizer.truncation_side = "left"
    encodings_full = emma_tokenizer.encode_plus(
        phrase, return_tensors="np", truncation=False
    ).input_ids[0]
    truncation_indices = [0] + list(range(-max_length + 1, 0))

    encodings_truncated = emma_tokenizer.encode_plus(
        phrase, return_tensors="np", truncation=True
    ).input_ids[0]
    assert all(encodings_truncated == encodings_full[truncation_indices])
