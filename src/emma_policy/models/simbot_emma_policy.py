import itertools
import json
import logging
from pathlib import Path
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
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.models.emma_policy import EmmaPolicy
from emma_policy.models.model_output_emma import EmmaSeq2SeqLMOutput


PredictType = Union[
    GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor
]
TrainBatchType = Union[EmmaDatasetBatch, dict[str, EmmaDatasetBatch]]
log = logging.getLogger(__name__)


class SimBotEmmaPolicy(EmmaPolicy):
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
        self._num_beams = num_beams
        self._min_length = 1
        self._max_generated_text_length = max_generated_text_length
        self._results_path = save_results_path
        self._example_ids: list[str] = []
        self._generated_actions: list[str] = []

    def on_test_epoch_end(self) -> None:
        """Save the results."""
        if self._results_path is not None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
                world_size = torch.distributed.get_world_size()
                all_example_ids = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(all_example_ids, self._example_ids)
                generated_actions = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(generated_actions, self._generated_actions)
                if torch.distributed.get_rank() == 0:
                    all_example_ids = list(
                        itertools.chain.from_iterable(all_example_ids)  # type: ignore[arg-type]
                    )
                    generated_actions = list(
                        itertools.chain.from_iterable(generated_actions)  # type: ignore[arg-type]
                    )
                    outputs = {
                        example_id: generated_action
                        for example_id, generated_action in zip(all_example_ids, generated_actions)
                    }
                    with open(self._results_path, "w") as fp:
                        json.dump(outputs, fp, indent=4)  # noqa: WPS220

            else:
                outputs = {
                    example_id: generated_action  # type: ignore[misc]
                    for example_id, generated_action in zip(
                        self._example_ids, self._generated_actions
                    )
                }
                with open(self._results_path, "w") as fp:  # noqa: WPS440
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
        max_length: int = 10,
        num_beams: int = 5,
        no_repeat_ngram_size: int = 0,
    ) -> PredictType:
        """Simbot Inference step."""
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
                max_length=max_length,
                decoder_input_ids=decoder_input_ids,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        else:
            outputs = self.emma.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.attention_mask,
                global_attention_mask=batch.global_attention_mask,
                decoder_encoder_attention_mask=batch.decoder_encoder_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        return outputs
