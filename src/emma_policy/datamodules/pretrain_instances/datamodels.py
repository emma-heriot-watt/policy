from enum import Enum
from types import MappingProxyType
from typing import Optional

from emma_datasets.datamodels import Instance, Region


class Task(Enum):
    """Task for the Pretraining instance."""

    mlm = "Masked Language Model"
    itm = "Image-Text Matching"
    visual_grounding = "Visual Grounding"
    dense_captioning = "Dense Captioning"
    captioning = "Captioning"
    vqa = "Visual Question Answering"
    instruction_prediction = "Instruction prediction for trajectory"
    action_execution = "Action execution in embodied environment"
    # TODO: check whether we can put together the entire trajectory
    # goal_prediction = "Goal prediction for an action trajectory"
    vtm = "Video-Text Matching"


TASK_PREFIX_MAP = MappingProxyType(
    {
        Task.mlm: "denoise",
        Task.itm: "evaluate statement",
        Task.visual_grounding: "select reference",
        Task.dense_captioning: "describe region",
        Task.captioning: "describe scene",
        Task.vqa: "answer question",
        Task.instruction_prediction: "predict instruction",
        Task.action_execution: "execute actions",
        # TODO: check whether we can put together the entire trajectory
        # Task.goal_prediction: "predict goal",
        Task.vtm: "evaluate statement",
    }
)


class PretrainInstance(Instance):  # type: ignore[misc]
    """Instance for the pretraining datamodule.

    This inherits and adds more attributes to the Instance from the emma_datasets library.
    """

    task: Task
    regions: Optional[Region]
