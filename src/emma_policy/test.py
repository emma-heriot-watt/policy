import warnings
from typing import Any

import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from emma_policy.utils import count_model_parameters, dump_config, get_logger


log = get_logger(__name__)


def verify_name_when_running_in_experiment_mode(config: DictConfig) -> None:
    """Verify experiment name is set when running in experiment mode."""
    if config.get("experiment_mode") and not config.get("name"):
        log.error("You must specify a name when running in experiment mode!")
        log.info("Exiting...")
        exit()  # noqa: WPS421


def debug_friendly_when_fast_dev_mode(config: DictConfig) -> None:
    """Force debugger friendly config if `config.fast_dev_run=True`.

    Debuggers don't like GPUs and multiprocessing, so disable all that.
    """
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


def ignore_warnings(config: DictConfig) -> None:
    """Disable python warnings if `config.ignore_warnings` is True."""
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")


class TrainModel:
    """Define a model from the config, proving interface for training.

    You should be using the `from_hydra_config` to instantiate the model. If you are calling the
    `__init__` function directly, something might not be right.
    """

    def __init__(
        self,
        config: DictConfig,
        datamodule: LightningDataModule,
        model: LightningModule,
        callbacks: list[Callback],
        loggers: list[LightningLoggerBase],
        trainer: Trainer,
    ) -> None:
        self.config = config
        self.datamodule = datamodule
        self.model = model
        self.callbacks = callbacks
        self.loggers = loggers
        self.trainer = trainer

        self.log_hyperparameters()

    def test(self) -> list[dict[str, float]]:
        """Test the model if config requires."""
        log.info("Testing...")
        scores = self.trainer.test(model=self.model, datamodule=self.datamodule)
        return scores

    def finish(self) -> None:
        """Finalise the model."""
        log.info("Finalising...")
        if any(isinstance(logger, WandbLogger) for logger in self.loggers):
            wandb.finish()

    @rank_zero_only
    def log_hyperparameters(self) -> None:
        """Send hyperparameters to all loggers."""
        model_parameters = count_model_parameters(self.model)

        hyperparams: dict[str, Any] = {
            "seed": self.config.get("seed"),
            "trainer": self.config.get("trainer"),
            "model": self.config.get("model"),
            "datamodule": self.config.get("datamodule"),
            "callbacks": self.config.get("callbacks"),
            "model/params/total": model_parameters.total,
            "model/params/trainable": model_parameters.trainable,
            "model/params/non_trainable": model_parameters.non_trainable,
        }

        if self.trainer.logger is not None:
            self.trainer.logger.log_hyperparams(hyperparams)  # type: ignore[arg-type]

    @classmethod
    def from_hydra_config(cls, config: DictConfig) -> "TrainModel":
        """Create class from hydra config.

        This should be the default method when implementing the model.
        """
        seed_everything(config.get("seed", None), workers=True)
        verify_name_when_running_in_experiment_mode(config)
        debug_friendly_when_fast_dev_mode(config)

        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")  # noqa: WPS437
        datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

        log.info(f"Instantiating model <{config.model._target_}>")  # noqa: WPS437
        model: LightningModule = hydra.utils.instantiate(config.model)
        if config.model.initialization_checkpoint is not None:
            model = model.load_from_checkpoint(config.model.initialization_checkpoint)

        callbacks: list[Callback] = []
        if "callbacks" in config:
            callback_configs = (
                callback for callback in config.callbacks.values() if "_target_" in callback
            )
            for callback in callback_configs:
                log.info(f"Instantiating callback <{callback._target_}>")  # noqa: WPS437
                callbacks.append(hydra.utils.instantiate(callback))

        loggers: list[LightningLoggerBase] = []
        if "logger" in config:
            logger_configs = (logger for logger in config.logger.values() if "_target_" in logger)
            for logger in logger_configs:
                log.info(f"Instantiating logger <{logger._target_}>")  # noqa: WPS437
                loggers.append(hydra.utils.instantiate(logger))

        log.info(f"Instantiating trainer <{config.trainer._target_}>")  # noqa: WPS437
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
        )

        return cls(config, datamodule, model, callbacks, loggers, trainer)


def test_model(config: DictConfig) -> list[dict[str, float]]:
    """Train model and return optimized metric."""
    ignore_warnings(config)

    if config.get("print_config"):
        dump_config(config)

    trainable_model = TrainModel.from_hydra_config(config)

    scores = trainable_model.test()
    trainable_model.finish()

    return scores
