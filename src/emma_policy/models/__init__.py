from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from emma_policy.models.configuration_emma import EmmaConfig
from emma_policy.models.modeling_emma import EmmaModel
from emma_policy.models.seq_emma import EmmaForConditionalGeneration
from emma_policy.models.tokenizer_emma import EmmaTokenizer
from emma_policy.models.tokenizer_emma_fast import EmmaTokenizerFast


AutoConfig.register("emma", EmmaConfig)
AutoTokenizer.register(
    EmmaConfig,  # type: ignore[arg-type]
    slow_tokenizer_class=EmmaTokenizer,
    fast_tokenizer_class=EmmaTokenizerFast,
)
AutoModel.register(EmmaConfig, EmmaModel)
AutoModelForCausalLM.register(EmmaConfig, EmmaForConditionalGeneration)
