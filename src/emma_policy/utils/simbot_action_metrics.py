from typing import Any, Union, cast

import torch
from overrides import overrides
from torchmetrics import Metric


class SimbotActionExactMatch(Metric):
    """Performs exact match at the sequence length.

    This means that we have a match if and only if *all* the elements of a sequence match. Masked
    elements are ignored if an appropriate mask tensor is specified.
    """

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(
        self, predicted: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Update loss sum and number of task samples."""
        # first compute all the matches ignoring padding
        masked_matches = (predicted == ground_truth) * mask
        # then make sure that the number of matched elements is equal to the actual sequence length
        num_matches_per_batch = masked_matches.sum(-1)
        seq_length_per_batch = mask.sum(-1)
        correct_matches = (num_matches_per_batch == seq_length_per_batch).int()
        self.correct += correct_matches.sum()
        self.total += cast(torch.Tensor, masked_matches.size(0))

    def compute(self) -> Union[torch.Tensor, Any]:
        """Compute the total task loss."""
        return self.correct.float() / self.total  # type: ignore[operator]
