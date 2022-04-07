import pytest
from pytest_cases import fixture, parametrize_with_cases

from emma_policy.inference import (
    TEACH_ACTION_TO_SYNONYMS,
    AgentAction,
    DecodedTrajectoryParser,
    get_synonyms_to_teach_action_map,
)
from emma_policy.models.tokenizer_emma import EmmaTokenizer


@fixture(scope="module")
def action_delimiter(emma_tokenizer: EmmaTokenizer) -> str:
    return emma_tokenizer.sep_token


class DecodedTeachTrajectories:
    """Various cases to ensure the TEACh trajectories are parsed correctly."""

    def case_forward(self) -> tuple[str, list[AgentAction]]:
        trajectory = "forward . move ahead . turn left ."
        api_actions = [
            AgentAction("Forward"),
            AgentAction("Forward"),
            AgentAction("Turn Left"),
        ]
        return trajectory, api_actions

    def case_interaction_object_and_vis_token(self) -> tuple[str, list[AgentAction]]:
        trajectory = "forward . move ahead . turn left . pick up mug <vis_token_3> ."
        api_actions = [
            AgentAction("Forward"),
            AgentAction("Forward"),
            AgentAction("Turn Left"),
            AgentAction(  # noqa: S106
                "Pickup", object_label="mug", object_visual_token="<vis_token_3>"
            ),
        ]
        return trajectory, api_actions

    def case_interaction_object_and_no_vis_token(self) -> tuple[str, list[AgentAction]]:
        trajectory = "forward . move ahead . turn left . pick up mug ."
        api_actions = [
            AgentAction("Forward"),
            AgentAction("Forward"),
            AgentAction("Turn Left"),
            AgentAction("Pickup", object_label="mug", object_visual_token=None),
        ]
        return trajectory, api_actions

    @pytest.mark.skip(reason="We assume that this case is not possible.")
    def case_only_interaction_visual_token(self) -> tuple[str, list[AgentAction]]:
        trajectory = "forward . move ahead . turn left . pick up <vis_token_3> ."
        api_actions = [
            AgentAction("Forward"),
            AgentAction("Forward"),
            AgentAction("Turn Left"),
            AgentAction(  # noqa: S106
                "Pickup", object_label=None, object_visual_token="<vis_token_3>"
            ),
        ]
        return trajectory, api_actions


@parametrize_with_cases("decoded_actions,expected_output", cases=DecodedTeachTrajectories)
def test_decoded_action_trajectories_are_converted_properly(
    decoded_actions: str, expected_output: list[AgentAction], action_delimiter: str
) -> None:
    trajectory_parser = DecodedTrajectoryParser(
        execution_domain="TEACh", action_delimiter=action_delimiter
    )
    parsed_trajectory = trajectory_parser(decoded_actions)

    assert parsed_trajectory == expected_output


def test_all_synonyms_are_mapped_to_teach_actions() -> None:
    """Ensure that each synonym is correctly mapped to one of the TEACh actions.

    Count the total number of synonyms across the mapping, and ensure that the count is identical
    to the size of the converted map.
    """
    total_synonyms_count = sum(
        len(synonym_set) for synonym_set in TEACH_ACTION_TO_SYNONYMS.values()
    )

    synonyms_actions_map = get_synonyms_to_teach_action_map()

    assert len(synonyms_actions_map) == total_synonyms_count
