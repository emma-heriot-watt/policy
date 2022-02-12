import string
from enum import Enum
from types import MappingProxyType
from typing import Iterator, Mapping, Optional

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


TASK_TEMPLATES_MAP: Mapping[Task, list[str]] = MappingProxyType(
    {
        Task.mlm: [
            "denoise: {caption}",
            "reconstruct the text: {caption}",
            "replace masked tokens: {caption}",
        ],
        Task.itm: ["evaluate statement: {statement}", "{statement}. is this statement coherent?"],
        Task.visual_grounding: [
            "which region can be described as {caption}",
            "select visual reference that matches {caption}",
        ],
        Task.dense_captioning: ["describe region {region}"],
        Task.captioning: ["describe the scene", "describe what you see"],
        Task.vqa: [
            "answer question: {question}",
            "what is the answer to the question {question}",
            "{question}",
        ],
        Task.instruction_prediction: ["predict instruction"],
        Task.action_execution: [
            "follow instruction: {instruction}",
            "complete task: {instruction}",
        ],
        # TODO: check whether we can put together the entire trajectory
        # Task.goal_prediction: "predict goal",
        Task.vtm: ["evaluate statement: {statement}", "{statement}. is this statement coherent?"],
    }
)


def extract_task_prefix_strings(templates_map: Mapping[Task, list[str]]) -> Iterator[str]:
    """Generates the string representation associated with each task template."""
    for templates in templates_map.values():
        for template in templates:
            empty_params: dict[str, str] = {
                name: "" for _, name, _, _ in string.Formatter().parse(template)
            }

            yield template.format(**empty_params)


class PretrainInstance(Instance):  # type: ignore[misc]
    """Instance for the pretraining datamodule.

    This inherits and adds more attributes to the Instance from the emma_datasets library.
    """

    task: Task
    regions: Optional[Region]
