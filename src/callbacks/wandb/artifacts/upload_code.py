from pathlib import Path

import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from src.callbacks.wandb.base import WandbCallbackBase


class UploadCode(WandbCallbackBase):
    """Upload source code for the run to WandB as an artifact."""

    def __init__(self, code_dir: str) -> None:
        self.code_dir = code_dir

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Upload source code as an artifact."""
        logger = self.get_wandb_logger(trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        for path in Path(self.code_dir).resolve().rglob("*.py"):
            code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)
