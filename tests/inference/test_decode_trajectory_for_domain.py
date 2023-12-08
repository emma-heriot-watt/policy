from pytest_cases import fixture

from emma_policy.models.tokenizer_emma import EmmaTokenizer


@fixture(scope="module")
def action_delimiter(emma_tokenizer: EmmaTokenizer) -> str:
    return emma_tokenizer.sep_token
