from transformers import LEDTokenizer


PRETRAINED_VOCAB_FILES_MAP = {  # noqa: WPS407
    "vocab_file": {
        "heriot-watt/emma-small": "https://huggingface.co/allenai/led-base-16384/resolve/main/vocab.json",
        "heriot-watt/emma-base": "https://huggingface.co/allenai/led-base-16384/resolve/main/vocab.json",
        "heriot-watt/emma-large": "https://huggingface.co/allenai/led-base-16384/resolve/main/vocab.json",
    },
    "merges_file": {
        "heriot-watt/emma-small": "https://huggingface.co/allenai/led-base-16384/resolve/main/merges.txt",
        "heriot-watt/emma-base": "https://huggingface.co/allenai/led-base-16384/resolve/main/merges.txt",
        "heriot-watt/emma-large": "https://huggingface.co/allenai/led-base-16384/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "heriot-watt/emma-small": "https://huggingface.co/allenai/led-base-16384/resolve/main/tokenizer.json",
        "heriot-watt/emma-base": "https://huggingface.co/allenai/led-base-16384/resolve/main/tokenizer.json",
        "heriot-watt/emma-large": "https://huggingface.co/allenai/led-base-16384/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {  # noqa: WPS407
    "heriot-watt/emma-small": 1024,
    "heriot-watt/emma-base": 4096,
    "heriot-watt/emma-large": 16384,
}


class EmmaTokenizer(LEDTokenizer):
    """Constructs an EMMA Tokenizer.

    [`EmmaTokenizer`] is identical to [`LEDTokenizer`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece. Refer to superclass [`LEDTokenizer`] for usage examples
    and documentation concerning parameters.
    """

    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # type: ignore[assignment]
