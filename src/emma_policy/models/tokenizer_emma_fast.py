from transformers import LEDTokenizerFast

from emma_policy.models.tokenizer_emma import EmmaTokenizer


PRETRAINED_VOCAB_FILES_MAP = {  # noqa: WPS407
    "vocab_file": {
        "heriot-watt/emma-small": "heriot-watt/emma-small/vocab.json",
        "heriot-watt/emma-base": "heriot-watt/emma-base/vocab.json",
    },
    "merges_file": {
        "heriot-watt/emma-small": "heriot-watt/emma-small/merges.txt",
        "heriot-watt/emma-base": "heriot-watt/emma-base/merges.txt",
    },
    "tokenizer_file": {
        "heriot-watt/emma-small": "heriot-watt/emma-small/tokenizer.json",
        "heriot-watt/emma-base": "heriot-watt/emma-base/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {  # noqa: WPS407
    "heriot-watt/emma-small": 4096,
    "heriot-watt/emma-base": 16384,
}


class EmmaTokenizerFast(LEDTokenizerFast):
    """Construct a "fast" EMMA tokenizer (backed by HuggingFace's *tokenizers* library).

    [`EmmaTokenizerFast`] is identical to [`LEDTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece. Refer to superclass [`LEDTokenizerFast`] for usage
    examples and documentation concerning parameters.
    """

    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # type: ignore[assignment]
    slow_tokenizer_class = EmmaTokenizer  # type: ignore[assignment]
    truncation_side = "right"
