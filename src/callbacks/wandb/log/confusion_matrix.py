from typing import Any, Optional

import torch
import wandb
from plotly import figure_factory as ff, graph_objects as go
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn import metrics

from src.callbacks.wandb.base import WandbCallbackBase


class ConfusionMatrix(WandbCallbackBase):
    """Generate confusion matrix and send to WandB.

    Confusion matrix is generated every epoch. Class expected the validation step to return
    predictions and targets.
    """

    def __init__(self) -> None:
        self._predictions: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._ready = True

    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Do not allow callback to do anything during validation sanity checks."""
        self.ready = False

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Gather data from single batch."""
        if self.ready and isinstance(outputs, dict):
            predictions: Optional[torch.Tensor] = outputs.get("predictions")
            targets: Optional[torch.Tensor] = outputs.get("targets")

            if predictions is not None and targets is not None:
                self._predictions.append(predictions)
                self._targets.append(targets)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Generate confusion matrix."""
        if self.ready:
            logger = self.get_wandb_logger(trainer)
            experiment = logger.experiment

            figure = self._create_confusion_matrix()

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log(
                {f"confusion_matrix/{experiment.name}": wandb.Plotly(figure)}, commit=False
            )

            self._predictions.clear()
            self._targets.clear()

    def _create_confusion_matrix(self) -> go.Figure:
        predictions = torch.cat(self._predictions).cpu().numpy()
        targets = torch.cat(self._targets).cpu().numpy()

        confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=predictions)

        # TODO(amit): Include labels for the confusion matrix
        # https://github.com/emma-simbot/research-base/issues/23

        fig = ff.create_annotated_heatmap(
            confusion_matrix,
            # x=x,
            # y=y,
            annotation_text=confusion_matrix.tostring(),
            colorscale="Plotly3",
        )

        return fig
