from typing import Any, Union

import torch
from overrides import overrides
from torchmetrics import Metric

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.utils.boxes import Boxes, matched_pairwise_iou


class RefCOCOAccuracy(Metric):
    """Loss for a pretraining task."""

    def __init__(self, dist_sync_on_step: bool = True, threshold: float = 0.5) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.threshold = threshold

    @overrides(check_signature=False)
    def update(self, outputs: torch.Tensor, batch: EmmaDatasetBatch) -> None:
        """Update loss sum and number of task samples."""
        indices = torch.where(batch.visual_token_ids == outputs[:, None])
        if indices[0].shape[0] > 0:
            predicted_bbox_coords = batch.object_coordinates[indices]

            predicted_bbox = Boxes(predicted_bbox_coords)
            # Stack the groundtruth boxes and keep the ones with valid visual token prediction
            target_bbox = Boxes(torch.stack(batch.raw_target)[indices[0]])  # type: ignore[arg-type]
            ious = matched_pairwise_iou(predicted_bbox, target_bbox)

            self.correct += torch.sum(ious >= self.threshold)
        # Divide with all examples in the batch, not only with the valid visual token prediction
        self.total += outputs.shape[0]  # type: ignore[operator]

    def compute(self) -> Union[torch.Tensor, Any]:
        """Compute the total task loss."""
        return self.correct.float() / self.total  # type: ignore[operator]
