from typing import Any, Union

import pytorch_lightning as pl
from overrides import overrides
from torch.optim import AdamW
from transformers import AutoConfig, get_linear_schedule_with_warmup

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.models.model_output_emma import EmmaSeq2SeqLMOutput
from emma_policy.models.seq_emma import EmmaForConditionalGeneration


class EmmaPolicy(pl.LightningModule):
    """Emma Lightning Module."""

    def __init__(self, model_name: str, **kwargs: dict[str, Any]) -> None:
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        self.emma = EmmaForConditionalGeneration(config)

        self.save_hyperparameters()

        self.trainer: pl.Trainer  # type: ignore[assignment]

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if (  # noqa: WPS337
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches != 0
        ):
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
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
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = (
            self.trainer.accumulate_grad_batches * num_devices  # type: ignore[attr-defined]
        )
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def compute_warmup(
        self, num_training_steps: int, num_warmup_steps: Union[int, float]
    ) -> tuple[int, int]:
        """Compute the number of total training and warmup steps."""
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps  # type: ignore[assignment]
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
                self.num_training_steps(),
                self.hparams.num_warmup_steps,  # type: ignore[union-attr]
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_training_steps, num_warmup_steps
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
            scene_frame_ids=batch.scene_frame_ids,
            object_features=batch.object_features,
            object_coordinates=batch.object_coordinates,
            object_frame_ids=batch.object_frame_ids,
            visual_token_ids=batch.visual_token_ids,
            language_token_ids=batch.input_token_ids,
            attention_mask=batch.attention_mask,
            global_attention_mask=batch.global_attention_mask,
            labels=batch.target_token_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
        )

        self.log("train_loss", output.loss)

        return output

    @overrides(check_signature=False)
    def validation_step(self, batch: EmmaDatasetBatch, batch_idx: int) -> EmmaSeq2SeqLMOutput:
        """Validation step."""
        output = self.emma(
            scene_features=batch.scene_features,
            scene_coordinates=batch.scene_coordinates,
            scene_frame_ids=batch.scene_frame_ids,
            object_features=batch.object_features,
            object_coordinates=batch.object_coordinates,
            object_frame_ids=batch.object_frame_ids,
            visual_token_ids=batch.visual_token_ids,
            language_token_ids=batch.input_token_ids,
            attention_mask=batch.attention_mask,
            global_attention_mask=batch.global_attention_mask,
            labels=batch.target_token_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
        )

        self.log("valid_loss", output.loss)

        return output