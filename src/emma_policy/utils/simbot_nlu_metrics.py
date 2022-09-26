from typing import Any, Union

import torch
from emma_datasets.datamodels.datasets.simbot import SimBotClarificationTypes
from overrides import overrides
from torchmetrics import ConfusionMatrix, F1Score, Metric
from transformers import PreTrainedTokenizer


class SimbotNLUExactMatch(Metric):
    """Loss for a exact match."""

    def __init__(self, dist_sync_on_step: bool = True, threshold: float = 0.5) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.threshold = threshold

    @overrides(check_signature=False)
    def update(self, predicted: list[str], ground_truth: list[list[str]]) -> None:
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
        self.class1_id = tokenizer.convert_tokens_to_ids("<act>")
        self.class2_id = tokenizer.convert_tokens_to_ids("<clarify>")
        super().__init__(num_classes=3, average="macro")

    @overrides(check_signature=False)
    def update(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> None:
        """Update loss sum and number of task samples."""
        # outputs begin with "</s><s>"
        act_predicted = torch.clone(predicted[:, 2])
        act_gt = torch.clone(ground_truth[:, 1])
        act_gt[act_gt == self.class1_id] = 0
        act_gt[act_gt == self.class2_id] = 1
        act_predicted[
            torch.logical_and(act_predicted != self.class1_id, act_predicted != self.class2_id)
        ] = 2
        act_predicted[act_predicted == self.class1_id] = 0
        act_predicted[act_predicted == self.class2_id] = 1
        super().update(act_predicted, act_gt)


class SimbotQuestionTypeConfusionMatrix(ConfusionMatrix):
    """Loss for a pretraining task."""

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.act_token_id = tokenizer.convert_tokens_to_ids("<act>")
        self.clarify_token_id = tokenizer.convert_tokens_to_ids("<clarify>")
        self.class_tokens = [
            f"<{qtype.name}>"
            for qtype in SimBotClarificationTypes
            if qtype != SimBotClarificationTypes.other
        ]
        class_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in self.class_tokens]
        self.class_token2id = {tokenid: idx for idx, tokenid in enumerate(class_token_ids)}

        super().__init__(num_classes=len(self.class_tokens), multilabel=True)

    @overrides(check_signature=False)
    def update(
        self, predicted: torch.Tensor, ground_truth_tensors: torch.Tensor, cm_targets: torch.Tensor
    ) -> None:
        """Update loss sum and number of task samples."""
        # outputs begin with "</s><s>"
        preds = torch.clone(predicted[ground_truth_tensors[:, 1] == self.clarify_token_id, 3])
        for token_id, class_index in self.class_token2id.items():
            preds[preds == token_id] = class_index
        preds[preds > self.num_classes] = 0
        cm_preds = torch.nn.functional.one_hot(preds.long(), num_classes=self.num_classes)
        cm_targets = cm_targets[ground_truth_tensors[:, 1] == self.clarify_token_id]
        super().update(cm_preds, cm_targets)
