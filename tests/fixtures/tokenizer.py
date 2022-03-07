from pytest_cases import fixture
from transformers import AutoTokenizer

from emma_policy.models.tokenizer_emma import EmmaTokenizer


@fixture(scope="module")
def emma_tokenizer(model_metadata_path: str) -> EmmaTokenizer:
    return AutoTokenizer.from_pretrained(model_metadata_path)
