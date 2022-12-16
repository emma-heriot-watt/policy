from typing import Any, Union

import torch
from overrides import overrides
from torchmetrics import F1Score, Metric
from transformers import PreTrainedTokenizer


class SimbotNLUExactMatch(Metric):
    """Loss for a exact match."""

    def __init__(self, dist_sync_on_step: bool = True, threshold: float = 0.5) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.threshold = threshold

    @overrides(check_signature=False)
    def update(self, predicted: list[str], ground_truth: list[str]) -> None:
        """Update loss sum and number of task samples."""
        for predicted_sentence, gt_sentences in zip(predicted, ground_truth):
            predicted_sentence = predicted_sentence.replace("</s>", "")
            predicted_sentence = predicted_sentence.replace("<pad>", "")
            predicted_sentence = predicted_sentence.replace("<s>", "")
            if predicted_sentence in gt_sentences:
                self.correct += 1  # type: ignore[operator]
        self.total += len(predicted)  # type: ignore[operator]

    def compute(self) -> Union[torch.Tensor, Any]:
        """Compute the total task loss."""
        return self.correct.float() / self.total  # type: ignore[operator]


class SimbotActionTypeF1(F1Score):
    """Loss for a pretraining task."""

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.class1_id = tokenizer.convert_tokens_to_ids("<one_match>")
        self.class2_id = tokenizer.convert_tokens_to_ids("<no_match>")
        self.class3_id = tokenizer.convert_tokens_to_ids("<too_many_matches>")
        super().__init__(num_classes=4, average="macro")

    @overrides(check_signature=False)
    def update(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> None:
        """Update loss sum and number of task samples."""
        # outputs begin with "</s><s>"
        act_predicted = torch.clone(predicted[:, 3])
        act_gt = torch.clone(ground_truth[:, 2])
        act_gt[act_gt == self.class1_id] = 0
        act_gt[act_gt == self.class2_id] = 1
        act_gt[act_gt == self.class3_id] = 2
        act_predicted[
            torch.logical_and(
                torch.logical_and(
                    act_predicted != self.class1_id,
                    act_predicted != self.class2_id,
                ),
                act_predicted != self.class3_id,
            )
        ] = 3
        act_predicted[act_predicted == self.class1_id] = 0
        act_predicted[act_predicted == self.class2_id] = 1
        act_predicted[act_predicted == self.class3_id] = 2
        super().update(act_predicted, act_gt)
