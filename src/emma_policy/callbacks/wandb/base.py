from typing import Optional

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase, LoggerCollection, WandbLogger


def _raise_error_if_fast_dev_run(trainer: Trainer) -> None:
    if trainer.fast_dev_run:
        raise RuntimeError(
            "Cannot use WandB Callbacks since Lighting disabled loggers in `fast_dev_run` mode."
        )


class WandbCallbackBase(Callback):
    """Base class for callbacks using WandbLogger."""

    def get_wandb_logger(self, trainer: Trainer) -> WandbLogger:
        """Safely get Weights & Biases logger from Trainer."""
        _raise_error_if_fast_dev_run(trainer)

        trainer_logger: Optional[LightningLoggerBase] = trainer.logger

        if isinstance(trainer_logger, WandbLogger):
            return trainer_logger

        if isinstance(trainer_logger, LoggerCollection):
            for logger in trainer_logger._logger_iterable:  # noqa: WPS437
                if isinstance(logger, WandbLogger):
                    return logger

        raise RuntimeError(
            "Currently using WandB-related callback but `WandbLogger` was not found."
        )
