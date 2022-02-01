from pathlib import Path

import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from src.callbacks.wandb.base import WandbCallbackBase


class UploadCheckpoints(WandbCallbackBase):
    """Upload model checkpoints for the run to WandB as an artifact."""

    def __init__(
        self, checkpoint_dir: str = "checkpoints/", only_upload_best: bool = False
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.only_upload_best = only_upload_best

    @rank_zero_only
    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: Exception
    ) -> None:
        """If interrupted, store checkpoints."""
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """When training finished, store checkpoints."""
        logger = self.get_wandb_logger(trainer)
        experiment = logger.experiment

        checkpoint_artifact = wandb.Artifact(name="experiment-checkpoints", type="checkpoints")

        if self.only_upload_best and trainer.checkpoint_callback is not None:
            checkpoint_artifact.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.checkpoint_dir).rglob("*.ckpt"):
                checkpoint_artifact.add_file(str(path))

        experiment.log_artifact(checkpoint_artifact)
