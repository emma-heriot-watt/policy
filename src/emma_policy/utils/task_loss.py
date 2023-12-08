from typing import Any, Union

import torch
from overrides import overrides
from torchmetrics import Metric


class TaskLoss(Metric):
    """Loss for a pretraining task."""

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("loss", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, losses: torch.Tensor, targets: torch.Tensor) -> None:
        """Update loss sum and number of task samples."""
        if losses.shape[0] > 0:
            self.loss += torch.sum(losses)
            self.total += torch.sum(targets > -1)

    def compute(self) -> Union[torch.Tensor, Any]:
        """Compute the total task loss."""
        return self.loss.float() / self.total  # type: ignore[operator]
