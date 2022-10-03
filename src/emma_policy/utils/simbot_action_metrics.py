from typing import Any, Union, cast

import torch
from overrides import overrides
from torchmetrics import Metric


class SimbotActionExactMatch(Metric):
    """Loss for a exact match."""

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(
        self, predicted: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Update loss sum and number of task samples."""
        masked_matches = (predicted == ground_truth) * mask
        masked_matches = masked_matches.sum(-1)
        self.correct += (masked_matches > 0).sum()
        self.total += cast(torch.Tensor, masked_matches.size(0))

    def compute(self) -> Union[torch.Tensor, Any]:
        """Compute the total task loss."""
        return self.correct.float() / self.total  # type: ignore[operator]
