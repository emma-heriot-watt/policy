from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from src.callbacks.wandb.base import WandbCallbackBase


class WatchModel(WandbCallbackBase):
    """Make WandB watch model at the beginning of the run."""

    def __init__(
        self, log: str = "gradients", log_freq: int = 100, log_graph: bool = False
    ) -> None:
        self.log = log
        self.log_freq = log_freq
        self.log_graph = log_graph

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Watch model on training start."""
        logger = self.get_wandb_logger(trainer)
        logger.watch(
            model=trainer.model,
            log=self.log,
            log_freq=self.log_freq,
            log_graph=self.log_graph,
        )
