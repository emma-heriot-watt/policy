import itertools
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
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.datamodules.simbot_combined_datamodule import prepare_combined_tokenizer
from emma_policy.models.emma_policy import EmmaPolicy
from emma_policy.models.model_output_emma import EmmaSeq2SeqLMOutput
from emma_policy.utils.simbot_action_metrics import SimbotActionExactMatch


PredictType = Union[
    GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor
]
TrainBatchType = Union[EmmaDatasetBatch, dict[str, EmmaDatasetBatch]]
log = logging.getLogger(__name__)


class SimBotEmmaCombinedPolicy(EmmaPolicy):
    """Emma Lightning Module."""

    def __init__(
        self,
        model_name: str,
        num_beams: int = 1,
        max_generated_text_length: int = 50,
        save_results_path: Optional[Path] = None,
        test_single_instance: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=f"{model_name}-combined", **kwargs)

        self.model_name = model_name
        self._tokenizer = prepare_combined_tokenizer(model_name)
        self._num_beams = num_beams
        self._min_length = 1
        self._max_generated_text_length = max_generated_text_length
        self._results_path = save_results_path
        self._example_ids: list[str] = []
        self._generated_actions: list[str] = []
        self._train_exact_match = SimbotActionExactMatch()
        self._valid_exact_match = SimbotActionExactMatch()
        self._separator_token_id = 370
        self._decoder_input_ids: list[str] = []
        self._decoder_confidences: list[float] = []
        self._groundtruth_actions: list[str] = []
        self._test_single_instance = test_single_instance

        # For the type of force_words_ids see: https://huggingfac.co/docs/transformers/v4.20.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.force_words_ids(List[List[int]]
        # force_words_ids lists will have lengths:
        # lengths number of constraints (1), number of visual tokens (100), number of tokens per forced word (1)
        force_words_ids = []
        ambiguity_words = [
            "<one_match>",
            "<no_match>",
            "<too_many_matches>",
            "<missing_inventory>",
        ]
        for action_word in ("<act>", "<search>"):
            for ambiguity_word in ambiguity_words:
                forced_phrase = [
                    self._tokenizer.convert_tokens_to_ids(action_word),
                    self._tokenizer.convert_tokens_to_ids(ambiguity_word),
                ]
                force_words_ids.append(forced_phrase)
        self.force_words_ids = [force_words_ids]
        if self._num_beams == 1:
            self._num_beams += 1  # constrains need num_beams > 1

    def on_test_epoch_end(self) -> None:  # noqa: WPS231
        """Save the results."""
        if self._test_single_instance and self._results_path is not None:
            all_example_ids = self._example_ids
            generated_actions = self._generated_actions
            groundtruth_actions = self._groundtruth_actions
            decoder_input_ids = self._decoder_input_ids

            outputs = {
                example_id: {
                    "prediction": generated_action,
                    "groundtruth": gt,
                    "teacher_forcing": dec,
                }
                for example_id, generated_action, gt, dec in zip(
                    all_example_ids,
                    generated_actions,
                    groundtruth_actions,
                    decoder_input_ids,
                )
            }
            self._save_results(outputs)

        elif self._results_path is not None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
                world_size = torch.distributed.get_world_size()
                all_example_ids = [None for _ in range(world_size)]  # type: ignore[misc]
                torch.distributed.all_gather_object(all_example_ids, self._example_ids)
                generated_actions = [None for _ in range(world_size)]  # type: ignore[misc]
                torch.distributed.all_gather_object(generated_actions, self._generated_actions)
                if torch.distributed.get_rank() == 0:
                    all_example_ids = list(itertools.chain.from_iterable(all_example_ids))
                    generated_actions = list(itertools.chain.from_iterable(generated_actions))
                    outputs = {
                        example_id: generated_action  # type: ignore[misc]
                        for example_id, generated_action in zip(all_example_ids, generated_actions)
                    }
                    self._save_results(outputs)
            else:
                outputs = {
                    example_id: generated_action  # type: ignore[misc]
                    for example_id, generated_action in zip(
                        self._example_ids, self._generated_actions
                    )
                }
                self._save_results(outputs)
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
            decoder_input_ids=batch.decoder_input_ids,
            decoder_encoder_attention_mask=batch.decoder_encoder_attention_mask,
        )

        self.log("train_loss", output.loss)

        predictions = torch.argmax(torch.softmax(output.logits, -1), -1)

        self._train_exact_match(predictions, batch.target_token_ids, batch.decoder_attention_mask)
        self.log("train_exact_match", self._train_exact_match)

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
            decoder_input_ids=batch.decoder_input_ids,
            decoder_encoder_attention_mask=batch.decoder_encoder_attention_mask,
        )

        predictions = torch.argmax(torch.softmax(output.logits, -1), -1)

        self._valid_exact_match(predictions, batch.target_token_ids, batch.decoder_attention_mask)

        self.log("valid_exact_match", self._valid_exact_match)

        self.log("valid_loss", output.loss, sync_dist=True)
        logits = output.logits
        targets = batch.target_token_ids.view(-1)
        loss_tasks = self.loss_fn(logits.view(-1, logits.shape[-1]), targets)
        output.losses = loss_tasks.view(logits.shape[0], -1)
        output.tasks = batch.task
        output.targets = batch.target_token_ids

        # log task loss on epoch
        for task_idx, task in enumerate(Task):
            data_indices = (output.tasks == task_idx).view(-1)
            self.task_metrics[task_idx](output.losses[data_indices], output.targets[data_indices])
            self.log(
                f"valid_{task.name}_loss",
                self.task_metrics[task_idx],
                on_epoch=True,
                on_step=False,
            )
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

    @overrides(check_signature=False)
    def test_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> PredictType:
        """Inference step."""
        if self._test_single_instance:
            return self._test_instance(batch)

        outputs = self.predict_step(batch=batch, batch_idx=batch_idx)
        outputs_text = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if batch.raw_target is not None:
            self._example_ids.extend(batch.raw_target)
        self._generated_actions.extend(outputs_text)
        return outputs

    @overrides(check_signature=False)
    def inference_step(
        self,
        batch: EmmaDatasetBatch,
        decoder_input_ids: Optional[torch.Tensor] = None,
        num_beams: int = 5,
        no_repeat_ngram_size: int = 0,
        use_force_word_ids: bool = False,
        max_length: Optional[int] = None,
    ) -> PredictType:
        """Simbot Inference step."""
        bad_words_ids = self._get_inference_banned_frame_ids(batch)
        max_generated_length = (
            max_length if max_length is not None else self._max_generated_text_length
        )

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

        if decoder_input_ids is not None:
            outputs = self.emma.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.attention_mask,
                global_attention_mask=batch.global_attention_mask,
                decoder_encoder_attention_mask=batch.decoder_encoder_attention_mask,
                max_length=max_generated_length,
                decoder_input_ids=decoder_input_ids,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,  # type: ignore[arg-type]
                force_words_ids=self.force_words_ids if use_force_word_ids else None,  # type: ignore[arg-type]
            )
        else:
            outputs = self.emma.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.attention_mask,
                global_attention_mask=batch.global_attention_mask,
                decoder_encoder_attention_mask=batch.decoder_encoder_attention_mask,
                max_length=max_generated_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,  # type: ignore[arg-type]
                force_words_ids=self.force_words_ids if use_force_word_ids else None,  # type: ignore[arg-type]
            )
        return outputs

    def _get_inference_banned_frame_ids(
        self, batch: EmmaDatasetBatch
    ) -> Optional[list[list[int]]]:
        """Prohibit decoding past frame ids during inference for a single instance."""
        bad_words_ids = None
        batch_size = batch.scene_features.shape[0]
        if batch_size > 1 or not isinstance(batch.raw_target, dict):
            return bad_words_ids

        minimum_frame_index = batch.raw_target.get("minimum_frame_index", 1)
        if minimum_frame_index > 1:
            bad_words_ids = [
                [self._tokenizer.convert_tokens_to_ids(f"<frame_token_{idx}>")]
                for idx in range(1, minimum_frame_index)
            ]
        return bad_words_ids  # type: ignore[return-value]

    def _test_instance(self, batch: EmmaDatasetBatch) -> PredictType:
        """Inference step."""
        if batch.decoder_input_ids is None:
            raise AssertionError("Expected decoder input ids for single instance testing")

        separator_positions = torch.where(batch.decoder_input_ids[0] == self._separator_token_id)[
            0
        ]

        if separator_positions.shape[0] > 1:
            end_index = int(separator_positions[-2].item()) + 1
            decoder_input_ids = batch.decoder_input_ids[:, :end_index]

        else:
            decoder_input_ids = batch.decoder_input_ids[:, 0].unsqueeze(0)

        outputs = self.inference_step(
            batch, decoder_input_ids=decoder_input_ids, max_length=self._max_generated_text_length
        )
        self._example_ids.append(batch.raw_target[0]["instance_id"])  # type: ignore[index]

        self._decoder_input_ids.extend(
            self._tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=False)
        )

        gt_text = []
        for target, prediction in zip(batch.target_token_ids, outputs):
            gt = target[target > -100]
            gt_text.append("".join(self._tokenizer.batch_decode(gt, skip_special_tokens=False)))
            pred = prediction[1:]
            self._generated_actions.append(
                "".join(self._tokenizer.batch_decode(pred, skip_special_tokens=False))
            )

        self._groundtruth_actions.extend(gt_text)
        return outputs

    def _save_results(self, outputs: dict[str, Any]) -> None:
        assert self._results_path is not None
        with open(self._results_path, "w") as fp:
            json.dump(outputs, fp, indent=4)
