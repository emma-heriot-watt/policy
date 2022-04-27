from typing import Any

import torch
from transformers.generation_stopping_criteria import StoppingCriteria


class ActionStopCriteria(StoppingCriteria):
    """This class can be used to stop generation.

    The generation stops whenever either these conditions hold:
        1. the last generated token is an action separator
        2. the last generated token is the eos token.
    """

    def __init__(self, action_sep_token_id: int, eos_token_id: int) -> None:
        self.action_sep_token_id = action_sep_token_id
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: dict[Any, Any]
    ) -> bool:
        """Compute the action separator condition after generating a single token."""
        action_sep_cond = torch.all(input_ids[:, -1] == self.action_sep_token_id).item()
        eos_cond = torch.all(input_ids[:, -1] == self.eos_token_id).item()
        return action_sep_cond or eos_cond  # type: ignore[return-value]
