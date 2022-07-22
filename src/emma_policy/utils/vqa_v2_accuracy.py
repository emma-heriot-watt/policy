from collections import Counter
from typing import Any, Union

import torch
from emma_datasets.datamodels.datasets.utils.vqa_v2_utils import normalize_answer
from emma_datasets.datamodels.datasets.vqa_v2 import vqa_v2_score
from overrides import overrides
from torchmetrics import Metric


class VQAv2Accuracy(Metric):
    """VQAv2 accuracy."""

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "accuracy", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, predicted_anwers: list[str], ground_truth_batch: list[list[str]]) -> None:
        """Update loss sum and number of task samples."""
        for predicted_answer, ground_truth_answers in zip(predicted_anwers, ground_truth_batch):
            predicted_answer = normalize_answer(predicted_answer)
            ground_truth_counts = Counter(ground_truth_answers)
            self.accuracy += torch.tensor(
                vqa_v2_score(ground_truth_counts.get(predicted_answer, 0))
            )

        self.total += torch.tensor(len(ground_truth_batch))

    def compute(self) -> Union[torch.Tensor, Any]:
        """Compute the total task loss."""
        return self.accuracy.float() / self.total  # type: ignore[operator]
