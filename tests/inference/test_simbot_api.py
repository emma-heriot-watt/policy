from typing import Optional

from pytest_cases import parametrize
from transformers import AutoTokenizer

from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.inference.api.simbot_state import GenerateRequest, RequestUtterance, SpeakerRole
from emma_policy.inference.model_wrapper.simbot_action_input_builder import (
    SimBotActionInputBuilder,
)


@parametrize(
    "input_request, target",
    [
        (
            GenerateRequest(
                dialogue_history=[RequestUtterance(role=SpeakerRole.user, utterance="")],
                environment_history=[],
            ),
            None,
        ),
        (
            GenerateRequest(
                dialogue_history=[
                    RequestUtterance(role=SpeakerRole.user, utterance="Instruction"),
                    RequestUtterance(role=SpeakerRole.agent, utterance="Is this a question?"),
                    RequestUtterance(role=SpeakerRole.user, utterance="Maybe"),
                ],
                environment_history=[],
            ),
            ("<<commander>> instruction. <<driver>> is this a question? <<commander>> maybe."),
        ),
        (
            GenerateRequest(
                dialogue_history=[
                    RequestUtterance(role=SpeakerRole.user, utterance="Instruction"),
                ],
                environment_history=[],
            ),
            "<<commander>> instruction.",
        ),
        (
            GenerateRequest(
                dialogue_history=[
                    RequestUtterance(role=SpeakerRole.user, utterance="Instruction1"),
                    RequestUtterance(role=SpeakerRole.agent, utterance="Is this a question?"),
                    RequestUtterance(role=SpeakerRole.user, utterance="Maybe"),
                    RequestUtterance(role=SpeakerRole.user, utterance="Instruction2"),
                ],
                environment_history=[],
            ),
            "<<commander>> instruction1. <<driver>> is this a question? <<commander>> maybe. <<commander>> instruction2.",
        ),
    ],
)
def test_simbot_action_builder_parses_dialogue_history(
    input_request: GenerateRequest,
    target: Optional[str],
) -> None:
    """Test that the action builder parses a request properly."""
    tokenizer = AutoTokenizer.from_pretrained("heriot-watt/emma-base")
    builder = SimBotActionInputBuilder(tokenizer=tokenizer)
    output = builder._parse_dialogue_from_request(input_request)
    assert output == target
    if output is not None:
        input_text = builder._prepare_input_text(instruction=output, task=Task.action_execution)
        assert input_text
