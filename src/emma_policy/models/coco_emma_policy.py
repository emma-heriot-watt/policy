import itertools
import logging
import os
from typing import Any, Optional, Union

import language_evaluation
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


PredictType = Union[
    GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor
]

log = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class COCOCaptioningEmmaPolicy(EmmaPolicy):
    """Emma Lightning Module."""

    def __init__(
        self,
        model_name: str,
        coco_metrics: Optional[list[str]] = None,
        num_beams: int = 5,
        max_text_length: int = 20,
        **kwargs: Any,
    ) -> None:
        self._captions: dict[str, list[str]] = {"predictions": [], "references": []}
        coco_metrics = ["BLEU", "CIDEr"] if coco_metrics is None else coco_metrics
        self._evaluator = language_evaluation.CocoEvaluator(coco_types=coco_metrics, verbose=False)
        self._num_beams = num_beams
        self._max_text_length = max_text_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        super().__init__(model_name=model_name, **kwargs)

    def on_validation_epoch_end(self) -> None:
        """Compute score and reset metrics after each validation epoch."""
        self._get_scores(mode="valid")
        self._reset_captions()
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        """Compute score and reset metrics after each test epoch."""
        self._get_scores(mode="test")
        self._reset_captions()
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
        prediction_output = self.predict_step(batch=batch, batch_idx=batch_idx)
        predictions = self._tokenizer.batch_decode(prediction_output, skip_special_tokens=True)
        references = batch.raw_target

        self._captions["predictions"].extend(predictions)
        self._captions["references"].extend(references)  # type: ignore[arg-type]

        return prediction_output

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

        output = self.emma.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.attention_mask,
            global_attention_mask=batch.global_attention_mask,
            max_length=self._max_text_length,
            num_beams=self._num_beams,
        )
        return output

    @overrides(check_signature=False)
    def test_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> PredictType:
        """Inference step."""
        prediction_output = self.predict_step(batch=batch, batch_idx=batch_idx)
        predictions = self._tokenizer.batch_decode(prediction_output, skip_special_tokens=True)
        references = batch.raw_target

        self._captions["predictions"].extend(predictions)
        self._captions["references"].extend(references)  # type: ignore[arg-type]

        return prediction_output

    def _compute_scores(
        self, predictions: list[str], references: list[str], mode: str = "valid"
    ) -> None:
        """Compute captioning scores."""
        eval_results = self._evaluator.run_evaluation(predictions, references)
        for metric, metric_score in eval_results.items():
            self.log(f"{mode}_{metric}", metric_score, sync_dist=False)

    def _reset_captions(self) -> None:
        """Reset the captions after computing the scores."""
        self._captions = {"predictions": [], "references": []}

    def _get_scores(self, mode: str = "valid") -> None:
        """Get scores."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            world_size = torch.distributed.get_world_size()
            output = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(output, self._captions)
            if torch.distributed.get_rank() == 0:
                self._captions["predictions"] = list(
                    itertools.chain.from_iterable([out["predictions"] for out in output])  # type: ignore[index]
                )
                self._captions["references"] = list(
                    itertools.chain.from_iterable([out["references"] for out in output])  # type: ignore[index]
                )
                self._compute_scores(
                    predictions=self._captions["predictions"],
                    references=self._captions["references"],
                    mode=mode,
                )
        else:
            self._compute_scores(
                predictions=self._captions["predictions"],
                references=self._captions["references"],
                mode=mode,
            )
