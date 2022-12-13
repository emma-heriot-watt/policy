import json
import logging
import re
from pathlib import Path
from typing import Optional

from emma_policy.inference.api.simbot_state import GenerateRequest, SpeakerRole
from emma_policy.utils.simbot_raw_text_matching import levenshtein_distance


logger = logging.getLogger(__name__)


class SimBotActionRawTextMatcher:
    """Simple raw text matcher used to minimise latency cost for trivial actions."""

    def __init__(self, raw_text_match_json: Path, distance_threshold: int = 2) -> None:
        with open(raw_text_match_json) as fp:
            self.raw_text_matching = json.load(fp)
        self.distance_threshold = distance_threshold

    def __call__(self, input_request: GenerateRequest) -> Optional[str]:
        """Process the input request."""
        if len(input_request.environment_history) > 1:
            logger.warning(
                "Received environment history for raw text match action prediction. This will be ignored."
            )

        if len(input_request.dialogue_history) >= 2:
            logger.warning(
                "Received multiple turns in the dialogue history. Only the first one will be considered."
            )

        request_utterance = input_request.dialogue_history[0]
        if request_utterance.role != SpeakerRole.user:
            logger.debug(
                f"The curret request does not have a user utterance: {input_request}. Returning None."
            )
            return None
        processed_str = self.preprocess_text(request_utterance.utterance)
        for action, action_metadata in self.raw_text_matching.items():
            action_templates = action_metadata["examples"]
            min_distance_for_action = min(
                [
                    levenshtein_distance(processed_str, action_template)
                    for action_template in action_templates
                ]
            )

            if min_distance_for_action < self.distance_threshold:
                output_string = self.postprocess_text(self.raw_text_matching[action]["command"])
                logger.debug(f"Matched input request to raw output action {output_string}.")
                return output_string
        logger.debug("Could not match input request to raw output action.")
        return None

    def preprocess_text(self, input_string: str) -> str:
        """Preprocess the raw input string."""
        new_string = re.sub(r"[^\w\s]", "", input_string)
        new_string = new_string.strip().lower()
        new_string = new_string.replace("can you", "")
        new_string = new_string.replace("can you please", "")
        new_string = new_string.replace("could you", "")
        new_string = new_string.replace("could you please", "")
        new_string = new_string.replace("please", "")
        new_string = " ".join(new_string.split())
        return new_string

    def postprocess_text(self, output_string: str) -> str:
        """Postprocess the output string.

        This should return a string that is suitable to handle by the experience hub.
        """
        return f"{output_string.lower()} <stop>.</s>"
