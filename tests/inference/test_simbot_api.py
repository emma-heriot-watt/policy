from typing import Optional

from pytest_cases import parametrize
from transformers import AutoTokenizer

from emma_policy.inference.api.simbot_state import GenerateRequest, RequestUtterance
from emma_policy.inference.model_wrapper.simbot_action_input_builder import (
    SimBotActionInputBuilder,
)


@parametrize(
    "input_request, target_tuple",
    [
        (
            GenerateRequest(
                dialogue_history=[
                    RequestUtterance(
                        role="user", utterance="User utterance", intent="clarify_answer"
                    )
                ],
                environment_history=[],
            ),
            (None, None, None),
        ),
        (
            GenerateRequest(
                dialogue_history=[
                    RequestUtterance(role="user", utterance="Instruction", intent="instruction"),
                    RequestUtterance(
                        role="agent", utterance="Is this a question?", intent="clarify_question"
                    ),
                    RequestUtterance(role="user", utterance="Maybe", intent="clarify_answer"),
                ],
                environment_history=[],
            ),
            ("Instruction", "Is this a question?", "Maybe"),
        ),
        (
            GenerateRequest(
                dialogue_history=[
                    RequestUtterance(role="user", utterance="Instruction", intent="instruction"),
                ],
                environment_history=[],
            ),
            ("Instruction", None, None),
        ),
        (
            GenerateRequest(
                dialogue_history=[
                    RequestUtterance(role="user", utterance="Instruction1", intent="instruction"),
                    RequestUtterance(
                        role="agent", utterance="Is this a question?", intent="clarify_question"
                    ),
                    RequestUtterance(role="user", utterance="Maybe", intent="clarify_answer"),
                    RequestUtterance(role="user", utterance="Instruction2", intent="instruction"),
                ],
                environment_history=[],
            ),
            ("Instruction2", None, None),
        ),
    ],
)
def test_simbot_action_builder_parses_dialogue_history(
    input_request: GenerateRequest,
    target_tuple: tuple[Optional[str], Optional[str], Optional[str]],
) -> None:
    """Test that the action builder parses a request properly."""
    tokenizer = AutoTokenizer.from_pretrained("heriot-watt/emma-base")
    builder = SimBotActionInputBuilder(tokenizer=tokenizer)
    output = builder._parse_dialogue_from_request(input_request)
    assert output == target_tuple
