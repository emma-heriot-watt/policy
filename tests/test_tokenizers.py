import logging

import pytest
from transformers import AutoTokenizer

from emma_policy.models.tokenizer_emma import EmmaTokenizer
from tests.conftest import TOKENIZER_PATHS


logger = logging.getLogger(__name__)


def load_tokenizers():
    return {
        AutoTokenizer.from_pretrained(tokenizer_path)
        for tokenizer_path in TOKENIZER_PATHS.values()
    }


@pytest.mark.parametrize("tokenizer", load_tokenizers())
def test_tokenize_input(tokenizer: EmmaTokenizer) -> None:
    phrase = "The dog is eating an icecream."
    encodings = tokenizer.encode_plus(phrase)

    logger.info(encodings)

    tokens = tokenizer.convert_ids_to_tokens(encodings.input_ids)
    conv_phrase = (
        tokenizer.convert_tokens_to_string(tokens)
        .replace(tokenizer.cls_token, "")
        .replace(tokenizer.eos_token, "")
    )
    assert phrase == conv_phrase
