import itertools
import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
from emma_datasets.io.json import write_json
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
from emma_policy.utils.vqa_v2_accuracy import VQAv2Accuracy


PredictType = Union[
    GreedySearchOutput,
    SampleOutput,
    BeamSearchOutput,
    BeamSampleOutput,
    torch.LongTensor,
    list[str],
]

log = logging.getLogger(__name__)


class VQAv2EmmaPolicy(EmmaPolicy):
    """Emma Lightning Module."""

    def __init__(
        self,
        model_name: str,
        num_beams: int = 1,
        max_generated_text_length: int = 20,
        save_results_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eval_acc = VQAv2Accuracy()
        self._num_beams = num_beams
        self._max_generated_text_length = max_generated_text_length
        self._min_length = 1
        self.task_metrics = None  # type: ignore[assignment]
        self._results_path = save_results_path
        self.question_ids: list[int] = []
        self.generated_answers: list[str] = []

    def on_test_epoch_end(self) -> None:
        """Save the QA resutls."""
        if self._results_path is not None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
                world_size = torch.distributed.get_world_size()
                all_question_ids = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(all_question_ids, self.question_ids)
                generated_answers = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(generated_answers, self.generated_answers)
                if torch.distributed.get_rank() == 0:
                    all_question_ids = list(
                        itertools.chain.from_iterable(all_question_ids)  # type: ignore[arg-type]
                    )
                    generated_answers = list(
                        itertools.chain.from_iterable(generated_answers)  # type: ignore[arg-type]
                    )
                    outputs = [
                        {"answer": answer, "question_id": qid}
                        for answer, qid in zip(generated_answers, all_question_ids)
                    ]
                    write_json(self._results_path, outputs)
            else:
                outputs = [
                    {"answer": answer, "question_id": qid}
                    for answer, qid in zip(generated_answers, all_question_ids)
                ]
                write_json(self._results_path, outputs)
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
    def validation_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> PredictType:
        """Validation step."""
        output = self._predict_and_compute_accuracy(batch=batch, batch_idx=batch_idx)
        self.log("valid_accuracy", self.eval_acc, on_step=False, on_epoch=True)
        return output

    @overrides(check_signature=False)
    def test_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> PredictType:
        """Inference step."""
        output = self._predict_answer(batch=batch, batch_idx=batch_idx)
        if batch.raw_target is not None:
            self.question_ids.extend([instance["question_id"] for instance in batch.raw_target])
        self.generated_answers.extend(output)
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
            decoder_encoder_attention_mask=batch.decoder_encoder_attention_mask,
            num_beams=self._num_beams,
            min_length=self._min_length,
            max_length=self._max_generated_text_length,
        )
        return outputs

    def _predict_answer(self, batch: EmmaDatasetBatch, batch_idx: int) -> list[str]:
        """Inference step."""
        output = self.predict_step(batch=batch, batch_idx=batch_idx)
        return self._tokenizer.batch_decode(output, skip_special_tokens=True)

    def _predict_and_compute_accuracy(
        self, batch: EmmaDatasetBatch, batch_idx: int
    ) -> PredictType:
        """Generate the bounding box and compute the accuracy."""
        predictions = self._predict_answer(batch, batch_idx)
        if batch.raw_target is not None:
            targets = [instance["answers"] for instance in batch.raw_target]
        self.eval_acc(predictions, targets)
        return predictions
