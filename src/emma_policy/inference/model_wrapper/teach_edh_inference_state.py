from typing import Optional

import torch

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem


class EdhInstanceInferenceState:
    """EDH Instance state used during inference."""

    def __init__(
        self,
        step_max_target_length: int,
        total_max_target_length: int,
        max_past_decoding_steps: int = 99,
        decoding_step: int = 1,
        decoding_input_ids: Optional[list[torch.Tensor]] = None,
        target_temporal_ids: Optional[list[torch.Tensor]] = None,
        eos_token_id: Optional[int] = 2,
    ) -> None:
        self.step_max_target_length = step_max_target_length
        self.total_max_target_length = total_max_target_length
        self.max_past_decoding_steps = max_past_decoding_steps
        self.decoding_step = decoding_step
        self._eos_token_id = eos_token_id
        self.decoding_input_ids: list[torch.Tensor] = (
            decoding_input_ids
            if decoding_input_ids is not None
            else [torch.tensor([self._eos_token_id], dtype=torch.int64)]
        )
        self.target_temporal_ids: list[torch.Tensor] = (
            target_temporal_ids
            if target_temporal_ids is not None
            else [torch.empty(0, dtype=torch.int64)]
        )

        self.reset_state()

    @property
    def previous_decoded_token_length(self) -> int:
        """Get the length of the previously decoded tokens."""
        return sum(len(tensor) for tensor in self.decoding_input_ids)

    @property
    def is_first_decoding_step(self) -> bool:
        """Return True if it is currently the first decoding step."""
        return self.decoding_step == 1

    def reset_state(self) -> None:
        """Reset the EDH state."""
        self.decoding_step = 1
        self.decoding_input_ids = [torch.tensor([self._eos_token_id], dtype=torch.int64)]
        self.target_temporal_ids = [torch.empty(0, dtype=torch.int64)]
        self.total_max_target_length = self.step_max_target_length

    def update_state(self, instance: EmmaDatasetItem, output_token_ids: torch.Tensor) -> None:
        """Update the state to prepare for the next prediction."""
        new_token_length = output_token_ids.shape[0] - self.previous_decoded_token_length
        # Fix the target token ids. Append to "past values" only the ones that were generated
        self.decoding_input_ids.append(output_token_ids[-new_token_length:])
        self.decoding_input_ids = self.decoding_input_ids[-self.max_past_decoding_steps :]

        # Fix the target temporal ids. Append the step number to the positions of the generated
        # output
        if instance.target_temporal_ids is not None:
            self.target_temporal_ids.append(
                torch.full(
                    size=(new_token_length,),
                    fill_value=self.decoding_step,
                    dtype=torch.int64,
                )
            )
            # Make sure that when truncating target temporal ids start from 1
            if len(self.target_temporal_ids) > self.max_past_decoding_steps:
                self.target_temporal_ids = self.target_temporal_ids[
                    -self.max_past_decoding_steps :
                ]
                for idx, temporal_ids in enumerate(self.target_temporal_ids, 1):
                    self.target_temporal_ids[idx - 1] = torch.full_like(temporal_ids, idx)

        # Increase the decoding step counter
        self.decoding_step = min(self.decoding_step + 1, self.max_past_decoding_steps + 1)
        # Update the total_max_target_length for the next decoding step
        self.total_max_target_length = output_token_ids.shape[0] + self.step_max_target_length
