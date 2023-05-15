from typing import Optional

from emma_common.datamodels import DialogueUtterance, EmmaPolicyRequest, SpeakerRole
from pytest_cases import parametrize
from transformers import AutoTokenizer

from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.inference.model_wrapper.simbot_action_input_builder import (
    SimBotActionInputBuilder,
)


@parametrize(
    "input_request, target",
    [
        (
            EmmaPolicyRequest(
                dialogue_history=[DialogueUtterance(role=SpeakerRole.user, utterance="")],
                environment_history=[],
            ),
            None,
        ),
        (
            EmmaPolicyRequest(
                dialogue_history=[
                    DialogueUtterance(role=SpeakerRole.user, utterance="Instruction"),
                    DialogueUtterance(role=SpeakerRole.agent, utterance="Is this a question?"),
                    DialogueUtterance(role=SpeakerRole.user, utterance="Maybe"),
                ],
                environment_history=[],
            ),
            ("<<commander>> instruction. <<driver>> is this a question? <<commander>> maybe."),
        ),
        (
            EmmaPolicyRequest(
                dialogue_history=[
                    DialogueUtterance(role=SpeakerRole.user, utterance="Instruction"),
                ],
                environment_history=[],
            ),
            "<<commander>> instruction.",
        ),
        (
            EmmaPolicyRequest(
                dialogue_history=[
                    DialogueUtterance(role=SpeakerRole.user, utterance="Instruction1"),
                    DialogueUtterance(role=SpeakerRole.agent, utterance="Is this a question?"),
                    DialogueUtterance(role=SpeakerRole.user, utterance="Maybe"),
                    DialogueUtterance(role=SpeakerRole.user, utterance="Instruction2"),
                ],
                environment_history=[],
            ),
            "<<commander>> instruction1. <<driver>> is this a question? <<commander>> maybe. <<commander>> instruction2.",
        ),
    ],
)
def test_simbot_action_builder_parses_dialogue_history(
    input_request: EmmaPolicyRequest,
    target: Optional[str],
) -> None:
    """Test that the action builder parses a request properly."""
    tokenizer = AutoTokenizer.from_pretrained("heriot-watt/emma-base")
    builder = SimBotActionInputBuilder(tokenizer=tokenizer)
    output = builder._parse_dialogue_from_request(input_request, task=Task.action_execution)
    assert output == target
    if output is not None:
        input_text = builder._prepare_input_text(instruction=output, task=Task.action_execution)
        assert input_text
