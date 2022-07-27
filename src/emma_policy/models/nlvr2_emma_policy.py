import itertools
import logging
from typing import Any, Optional, Union

import torch
from overrides import overrides
from transformers import AutoTokenizer
from transformers.generation_utils import (
    BeamSampleOutput,
    BeamSearchOutput,
    GreedySearchOutput,
    SampleOutput,
)

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.models.emma_policy import EmmaPolicy
from emma_policy.models.model_output_emma import EmmaSeq2SeqLMOutput
from emma_policy.utils.nlvr2_metrics import NLVR2Evaluator


PredictType = Union[
    GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor
]

log = logging.getLogger(__name__)


class NLVR2EmmaPolicy(EmmaPolicy):
    """Emma Lightning Module."""

    def __init__(
        self,
        model_name: str,
        max_text_length: int = 5,
        nlvr2_metrics: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._pred_gt: dict[str, list[str]] = {
            "predictions": [],
            "ground_truth": [],
            "sentences": [],
        }
        self._evaluator = NLVR2Evaluator()
        self._max_text_length = max_text_length
        self._nlvr2_metrics = nlvr2_metrics
        super().__init__(model_name=model_name, kwargs=kwargs)

    def on_validation_epoch_end(self) -> None:
        """Compute score and reset metrics after each validation epoch."""
        self._get_scores()
        self._reset_pred_gt()
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        """Compute score and reset metrics after each test epoch."""
        self._get_scores()
        self._reset_pred_gt()
        return super().on_test_epoch_end()

    @overrides(check_signature=False)
    def training_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> EmmaSeq2SeqLMOutput:
        """Training step."""
        output = self.emma(
            scene_features=batch.scene_features,
            scene_coordinates=batch.scene_coordinates,
            scene_frame_tokens=batch.scene_frame_tokens,
            object_features=batch.object_features,
            object_coordinates=batch.object_coordinates,
            object_frame_tokens=batch.object_frame_tokens,
            visual_token_ids=batch.visual_token_ids,
            language_token_ids=batch.input_token_ids,
            attention_mask=batch.attention_mask,
            global_attention_mask=batch.global_attention_mask,
            labels=batch.target_token_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            decoder_encoder_attention_mask=batch.decoder_encoder_attention_mask,
        )

        self.log("train_loss", output.loss)

        return output

    @overrides(check_signature=False)
    def validation_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> EmmaSeq2SeqLMOutput:
        """Validation step."""
        output = self.emma(
            scene_features=batch.scene_features,
            scene_coordinates=batch.scene_coordinates,
            scene_frame_tokens=batch.scene_frame_tokens,
            object_features=batch.object_features,
            object_coordinates=batch.object_coordinates,
            object_frame_tokens=batch.object_frame_tokens,
            visual_token_ids=batch.visual_token_ids,
            language_token_ids=batch.input_token_ids,
            attention_mask=batch.attention_mask,
            global_attention_mask=batch.global_attention_mask,
            labels=batch.target_token_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            decoder_encoder_attention_mask=batch.decoder_encoder_attention_mask,
        )

        prediction_output = self.predict_step(batch=batch, batch_idx=batch_idx)
        predictions = self._tokenizer.batch_decode(prediction_output, skip_special_tokens=True)
        batch.target_token_ids[batch.target_token_ids < 0] = self._tokenizer.pad_token_id
        ground_truth = self._tokenizer.batch_decode(
            batch.target_token_ids, skip_special_tokens=True
        )
        sentences = self._tokenizer.batch_decode(batch.input_token_ids, skip_special_tokens=True)

        self.log("valid_loss", output.loss)
        self._pred_gt["predictions"].extend(predictions)
        self._pred_gt["ground_truth"].extend(ground_truth)
        self._pred_gt["sentences"].extend(sentences)

        return output

    @overrides(check_signature=False)
    def predict_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> PredictType:
        """Inference step."""
        inputs_embeds = self.emma.emma.embed_inputs(
            scene_features=batch.scene_features,
            scene_coordinates=batch.scene_coordinates,
            scene_frame_tokens=batch.scene_frame_tokens,
            object_features=batch.object_features,
            object_coordinates=batch.object_coordinates,
            object_frame_tokens=batch.object_frame_tokens,
            visual_token_ids=batch.visual_token_ids,
            language_token_ids=batch.input_token_ids,
        )
        outputs = self.emma.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.attention_mask,
            global_attention_mask=batch.global_attention_mask,
            max_length=self._max_text_length,
            decoder_encoder_attention_mask=batch.decoder_encoder_attention_mask,
        )
        return outputs

    @overrides(check_signature=False)
    def test_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> PredictType:
        """Test step."""
        prediction_output = self.predict_step(batch=batch, batch_idx=batch_idx)
        predictions = self._tokenizer.batch_decode(prediction_output, skip_special_tokens=True)
        batch.target_token_ids[batch.target_token_ids < 0] = self._tokenizer.pad_token_id
        ground_truth = self._tokenizer.batch_decode(
            batch.target_token_ids, skip_special_tokens=True
        )
        sentences = self._tokenizer.batch_decode(batch.input_token_ids, skip_special_tokens=True)

        self._pred_gt["predictions"].extend(predictions)
        self._pred_gt["ground_truth"].extend(ground_truth)
        self._pred_gt["sentences"].extend(sentences)

        return prediction_output

    def _compute_scores(
        self, predictions: list[str], ground_truth: list[str], sentences: list[str]
    ) -> None:
        """Compute nlvr2 scores."""
        eval_results = self._evaluator.run_evaluation(predictions, ground_truth, sentences)
        split = "valid"
        if not self.trainer.validating:
            split = "test"
        for metric, metric_score in eval_results.items():
            self.log(f"{split}_{metric}", metric_score, sync_dist=False)

    def _reset_pred_gt(self) -> None:
        """Reset the pred_gt after computing the scores."""
        self._pred_gt = {"predictions": [], "ground_truth": [], "sentences": []}

    def _get_scores(self) -> None:
        """Get scores."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            world_size = torch.distributed.get_world_size()
            output = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(output, self._pred_gt)
            if torch.distributed.get_rank() == 0:
                self._pred_gt["predictions"] = list(
                    itertools.chain.from_iterable([out["predictions"] for out in output])  # type: ignore[index]
                )

                self._pred_gt["ground_truth"] = list(
                    itertools.chain.from_iterable([out["ground_truth"] for out in output])  # type: ignore[index]
                )

                self._pred_gt["sentences"] = list(
                    itertools.chain.from_iterable([out["sentences"] for out in output])  # type: ignore[index]
                )

                self._compute_scores(
                    predictions=self._pred_gt["predictions"],
                    ground_truth=self._pred_gt["ground_truth"],
                    sentences=self._pred_gt["sentences"],
                )
        else:
            self._compute_scores(
                predictions=self._pred_gt["predictions"],
                ground_truth=self._pred_gt["ground_truth"],
                sentences=self._pred_gt["sentences"],
            )
