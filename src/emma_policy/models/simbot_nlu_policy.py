import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
from overrides import overrides
from transformers.generation_utils import (
    BeamSampleOutput,
    BeamSearchOutput,
    GreedySearchOutput,
    SampleOutput,
)

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.simbot_nlu_datamodule import prepare_nlu_tokenizer
from emma_policy.models.emma_policy import EmmaPolicy
from emma_policy.models.model_output_emma import EmmaSeq2SeqLMOutput
from emma_policy.utils.simbot_nlu_metrics import SimbotActionTypeF1, SimbotNLUExactMatch


PredictType = Union[
    GreedySearchOutput,
    SampleOutput,
    BeamSearchOutput,
    BeamSampleOutput,
    torch.LongTensor,
]

log = logging.getLogger(__name__)
ForcedWordIdsList = list[list[list[int]]]


class SimBotNLUEmmaPolicy(EmmaPolicy):
    """Emma Lightning Module."""

    def __init__(
        self,
        model_name: str,
        num_beams: int = 5,
        max_generated_text_length: int = 16,
        save_results_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=f"{model_name}-nlu", **kwargs)
        self.model_name = model_name
        self._question_answers: dict[str, list[str]] = {"predictions": [], "references": []}

        self._num_beams = num_beams
        self._tokenizer = prepare_nlu_tokenizer(model_name=model_name)
        self._min_length = 1
        self._max_generated_text_length = max_generated_text_length

        # For the type of force_words_ids see: https://huggingfac.co/docs/transformers/v4.20.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.force_words_ids(List[List[int]]
        # force_words_ids lists will have lengths:
        # lengths number of constraints (1), number of visual tokens (100), number of tokens per forced word (1)
        action = [
            [self._tokenizer.convert_tokens_to_ids(forced_word)]
            for forced_word in ("<act>", "<search>")
        ]

        ambiguity = [
            [self._tokenizer.convert_tokens_to_ids(forced_word)]
            for forced_word in ("<one_match>", "<no_match>", "<too_many_matches>")
        ]
        self.force_words_ids = [action, ambiguity]
        if self._num_beams == 1:
            self._num_beams += 1  # constrains need num_beams > 1
        self.task_metrics = None  # type: ignore[assignment]
        self.validation_action_type_F1 = SimbotActionTypeF1(tokenizer=self._tokenizer)
        self.validation_accuracy = SimbotNLUExactMatch()

        self._results_path = save_results_path
        self._test_results: dict[str, list[str]] = {
            "example_ids": [],
            "instructions": [],
            "predictions": [],
            "groundtruths": [],
        }

    def on_test_epoch_end(self) -> None:
        """Save predictions at the end of the evaluation."""
        if self._results_path is None:
            return super().on_test_epoch_end()
        all_example_ids = self._test_results["example_ids"]
        instructions = self._test_results["instructions"]
        groundtruth_actions = self._test_results["groundtruths"]
        predictions = self._test_results["predictions"]
        outputs = {
            example_id: {"input": instr, "prediction": pred, "groundtruth": gt}
            for example_id, pred, gt, instr in zip(
                all_example_ids, predictions, groundtruth_actions, instructions
            )
        }
        with open(self._results_path, "w") as fp:
            json.dump(outputs, fp, indent=4)
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
        self.compute_metrics(prediction_output, batch)
        return prediction_output

    @overrides(check_signature=False)
    def test_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> None:
        """Inference step."""
        prediction_output = self.predict_step(batch=batch, batch_idx=batch_idx)
        # Prepare the results
        if self._results_path:
            self._test_results["example_ids"].extend(
                [sample["example_id"] for sample in batch.raw_target]  # type: ignore[union-attr]
            )
            self._test_results["instructions"].extend(
                [sample["instruction"] for sample in batch.raw_target]  # type: ignore[union-attr]
            )
            self._test_results["groundtruths"].extend(
                [sample["references"] for sample in batch.raw_target]  # type: ignore[union-attr]
            )
            for sent in self._tokenizer.batch_decode(prediction_output, skip_special_tokens=False):
                sent = self._remove_sequence_special_tokens(sent)
                self._test_results["predictions"].append(sent)

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
            max_length=self._max_generated_text_length,
            num_beams=self._num_beams,
            force_words_ids=self.force_words_ids,  # type: ignore[arg-type]
            min_length=self._min_length,
        )
        return output

    @overrides(check_signature=False)
    def inference_step(self, batch: EmmaDatasetBatch, batch_idx: int = 0) -> list[str]:
        """Inference step."""
        output_tokens = self.predict_step(batch, batch_idx)
        return self._postprocess_nlu_output(output_tokens)

    def compute_metrics(self, prediction_output: torch.Tensor, batch: EmmaDatasetBatch) -> None:
        """Compute the evaluation metrics."""
        self.validation_action_type_F1(prediction_output, batch.target_token_ids)
        self.log("validation_action_type_F1", self.validation_action_type_F1)
        predictions = self._tokenizer.batch_decode(prediction_output, skip_special_tokens=False)
        references = [
            sample["references"] for sample in batch.raw_target  # type: ignore[union-attr]
        ]
        self.validation_accuracy(predictions, references)
        self.log("validation_accuracy", self.validation_accuracy)

    def _postprocess_nlu_output(self, output: list[str]) -> list[str]:
        """Remove special tokens from predicted outputs."""
        output_sentences = self._tokenizer.batch_decode(output, skip_special_tokens=False)
        output_sentences = [
            self._remove_sequence_special_tokens(sentence) for sentence in output_sentences
        ]
        return output_sentences

    def _remove_sequence_special_tokens(self, sentence: str) -> str:
        """Remove the start, end of sequence and padding tokens from a string."""
        special_tokens = [
            self._tokenizer.bos_token,
            self._tokenizer.eos_token,
            self._tokenizer.pad_token,
        ]
        for special_token in special_tokens:
            sentence = sentence.replace(special_token, "")
        return sentence
