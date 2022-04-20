import logging
from typing import Any, Union

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoConfig, get_linear_schedule_with_warmup
from transformers.generation_utils import (
    BeamSampleOutput,
    BeamSearchOutput,
    GreedySearchOutput,
    SampleOutput,
)

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.models.model_output_emma import EmmaSeq2SeqLMOutput
from emma_policy.models.seq_emma import EmmaForConditionalGeneration
from emma_policy.utils.task_loss import TaskLoss


PredictType = Union[
    GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor
]

log = logging.getLogger(__name__)


class EmmaPolicy(pl.LightningModule):
    """Emma Lightning Module."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        self.emma = EmmaForConditionalGeneration(config)
        self.loss_fn = CrossEntropyLoss(reduction="none")

        self.save_hyperparameters()
        self.task_metrics = torch.nn.ModuleList([TaskLoss() for _ in Task])

        self.trainer: pl.Trainer  # type: ignore[assignment]

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if (  # noqa: WPS337
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches != 0
        ):
            dataset_size = self.trainer.limit_train_batches
            num_devices = 1
        else:
            if isinstance(self.trainer.limit_train_batches, float):
                # limit_train_batches is a percentage of batches
                dataset_size = len(
                    self.trainer.datamodule.train_dataloader()  # type: ignore[attr-defined]
                )
                dataset_size = int(dataset_size * self.trainer.limit_train_batches)
            else:
                dataset_size = len(  # type: ignore[unreachable]
                    self.trainer.datamodule.train_dataloader()
                )
            num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)

        # Check if using tpus
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = (
            self.trainer.accumulate_grad_batches * num_devices  # type: ignore[attr-defined]
        )
        num_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < num_steps:
            num_steps = self.trainer.max_steps

        log.info(f"Total number of training steps: {num_steps}")

        return num_steps

    def compute_warmup(
        self, num_training_steps: int, num_warmup_steps: Union[int, float]
    ) -> tuple[int, int]:
        """Compute the number of total training and warmup steps."""
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps()
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, int(num_warmup_steps)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer and learning rate scheduler."""
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [
                    param_value
                    for param_name, param_value in self.named_parameters()
                    if not any(nd in param_name for nd in no_decay) and param_value.requires_grad
                ],
                "weight_decay": self.hparams.weight_decay,  # type: ignore[union-attr]
            },
            {
                "params": [
                    param_value
                    for param_name, param_value in self.named_parameters()
                    if any(nd in param_name for nd in no_decay) and param_value.requires_grad
                ],
                "weight_decay": 0,
            },
        ]

        if self.hparams.optimizer == "adamw":  # type: ignore[union-attr]
            optimizer = AdamW(
                grouped_parameters,
                lr=self.hparams.lr,  # type: ignore[union-attr]
                weight_decay=self.hparams.weight_decay,  # type: ignore[union-attr]
            )
        else:
            raise ValueError("Invalid optimizer option")

        if self.hparams.lr_scheduler == "linear_with_warmup":  # type: ignore[union-attr]
            num_training_steps, num_warmup_steps = self.compute_warmup(
                self.trainer.max_steps,
                self.hparams.num_warmup_steps,  # type: ignore[union-attr]
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps
            )
        else:
            raise ValueError("Invalid optimizer option")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

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
        )

        return outputs
