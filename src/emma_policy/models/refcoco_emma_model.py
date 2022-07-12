import logging
from typing import Any, Optional, Union

import torch
from overrides import overrides
from torchmetrics import Accuracy
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

ForcedWordIdsList = list[list[list[int]]]


class RefCocoEmmaPolicy(EmmaPolicy):
    """Emma Lightning Module."""

    def __init__(
        self,
        model_name: str,
        num_beams: int = 1,
        max_generated_text_length: int = 4,
        constrain_outputs: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_acc = Accuracy()
        self.eval_acc = Accuracy()
        self._num_beams = num_beams
        self._max_generated_text_length = max_generated_text_length
        self._min_length = 1
        # For the type of force_words_ids see: https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.force_words_ids(List[List[int]]
        # force_words_ids lists will have lengths:
        # lengths number of constraints (1), number of visual tokens (100), number of tokens per forced word (1)
        self.force_words_ids: Optional[ForcedWordIdsList] = None
        if constrain_outputs:
            self.force_words_ids = [
                [
                    [token_id]
                    for token_id in self._tokenizer.additional_special_tokens_ids
                    if self._tokenizer.decode(token_id).startswith("<vis_token_")
                ]
            ]
            if self._num_beams == 1:
                self._num_beams += 1  # constrains need num_beams > 1

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
        self.train_acc(output.logits[:, 1, :], batch.target_token_ids[:, 1])
        self.log("train_accuracy", self.train_acc, on_step=True, on_epoch=False)

        return output

    @overrides(check_signature=False)
    def validation_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> PredictType:
        """Validation step."""
        output = self._predict_and_compute_accuracy(batch=batch, batch_idx=batch_idx)
        self.log("valid_accuracy", self.eval_acc, on_step=True, on_epoch=True)
        return output

    @overrides(check_signature=False)
    def test_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> PredictType:
        """Inference step."""
        output = self._predict_and_compute_accuracy(batch=batch, batch_idx=batch_idx)
        self.log("test_accuracy", self.eval_acc, on_step=True, on_epoch=True)
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
            force_words_ids=self.force_words_ids,  # type: ignore[arg-type]
            num_beams=self._num_beams,
            min_length=self._min_length,
            max_length=self._max_generated_text_length,
        )
        return outputs

    def _predict_and_compute_accuracy(
        self, batch: EmmaDatasetBatch, batch_idx: int
    ) -> PredictType:
        """Generate the bounding box and compute the accuracy."""
        output = self.predict_step(batch=batch, batch_idx=batch_idx)
        # outputs begin with "</s><s>"
        self.eval_acc(output[:, 2], batch.target_token_ids[:, 1])
        return output
