from pytest_cases import fixture
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from emma_policy.models.emma_policy import EmmaPolicy


@fixture
def emma_model_for_causal_lm(model_metadata_path: str) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_metadata_path)
    model = AutoModelForCausalLM.from_config(config)
    return model


@fixture
def emma_policy_model(model_metadata_path: str) -> EmmaPolicy:
    policy_model = EmmaPolicy(
        model_metadata_path,
        lr=0.0001,
        weight_decay=0.01,
        optimizer="adamw",
        lr_scheduler="linear_with_warmup",
        num_warmup_steps=0.05,
    )
    return policy_model
