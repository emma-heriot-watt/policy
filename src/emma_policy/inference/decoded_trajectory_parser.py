import logging
from typing import Literal

from emma_policy.inference.actions import (
    AgentAction,
    get_lowercase_to_teach_object_map,
    get_synonyms_to_teach_action_map,
)


logger = logging.getLogger(__name__)


ExecutionDomain = Literal["TEACh", "AI2THOR"]


class DecodedTrajectoryParser:
    """Convert the decoded action trajectory from the model to execute on the given domain."""

    def __init__(
        self,
        execution_domain: ExecutionDomain,
        action_delimiter: str,
        eos_token: str,
    ) -> None:
        self._execution_domain = execution_domain
        self._action_delimiter = action_delimiter

        self._synonym_to_action_map = get_synonyms_to_teach_action_map()
        self.eos_token = eos_token
        self.lowercase_to_teach_objects = get_lowercase_to_teach_object_map()

    def __call__(self, decoded_trajectory: str) -> AgentAction:
        """Converts a sequence of tokens into a list of executable actions."""
        logger.debug(f"Decoded trajectory: `{decoded_trajectory}`")

        if decoded_trajectory.endswith(self.eos_token):
            return AgentAction(action="Stop")

        decoded_actions_list = self._separate_decoded_trajectory(decoded_trajectory)
        return self._convert_action_to_executable_form(decoded_actions_list[0])

    def _separate_decoded_trajectory(self, decoded_trajectory: str) -> list[str]:
        """Split the decoded trajectory string into a list of action strings.

        Uses the given action delimiter (which is likely going to be the tokenizer SEP token).

        Also removes any blank strings from the list of actions.
        """
        split_actions = decoded_trajectory.split(self._action_delimiter)
        return [action for action in split_actions if action]

    def _get_teach_action_from_tokens(self, action_tokens: list[str]) -> tuple[str, list[str]]:
        """Get the teach action from the decoded action string.

        Assumptions:
            - The action appears at the start of the `decoded_action_string`.
            - The action can be of a length more than 1.

        Example:
            - If decoded_action == `forward`, then return `Forward`
            - If decoded_action == `pickup mug`, then return `Pickup`
        """
        action_name = None

        index = len(action_tokens)
        while index > 0:
            action_name = " ".join(action_tokens[:index])

            if action_name in self._synonym_to_action_map:
                break

            index -= 1

        if action_name is None:
            # edge case: we were not able to map the current action, just return an empty action
            return "", action_tokens

        return self._synonym_to_action_map[action_name], action_tokens[index:]

    def _convert_action_to_executable_form(self, action_str: str) -> AgentAction:
        """Convert the decoded action string into an executable form.

        We need to handle different cases:
            - Index 0: Should be the TEACh API Action
            - Index 1: Should be the object class (when available)
            - Index 2: Should be the visual token (when available)

        We are assuming that the visual token will only ever be present after the object class.
        """
        action_tokens = action_str.strip().split(" ")

        teach_action, teach_action_params = self._get_teach_action_from_tokens(action_tokens)

        object_label = None
        object_visual_token = None
        raw_object_label = None

        for action_param in teach_action_params:
            action_param = action_param.strip()

            if action_param.startswith("<") and action_param.endswith(">"):
                object_visual_token = action_param.strip()
            else:
                obj_label = self.lowercase_to_teach_objects.get(action_param)
                if obj_label is not None:
                    object_label = self.lowercase_to_teach_objects[action_param]
                else:
                    raw_object_label = action_param

        return AgentAction(
            action=teach_action,
            object_label=object_label,
            raw_object_label=raw_object_label,
            object_visual_token=object_visual_token,
        )
