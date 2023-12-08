from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from emma_policy._version import __version__
from emma_policy.models import (
    EmmaConfig,
    EmmaForConditionalGeneration,
    EmmaModel,
    EmmaTokenizer,
    EmmaTokenizerFast,
)
from emma_policy.test import test_model
from emma_policy.train import train_model


AutoConfig.register("emma", EmmaConfig)
AutoTokenizer.register(
    EmmaConfig,  # type: ignore[arg-type]
    slow_tokenizer_class=EmmaTokenizer,
    fast_tokenizer_class=EmmaTokenizerFast,
)
AutoModel.register(EmmaConfig, EmmaModel)
AutoModelForCausalLM.register(EmmaConfig, EmmaForConditionalGeneration)
