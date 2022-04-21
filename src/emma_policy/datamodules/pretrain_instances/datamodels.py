import string
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional, Union, cast

from emma_datasets.datamodels import Caption, MediaType, QuestionAnswerPair, Region
from emma_datasets.datamodels.instance import ActionTrajectory, MultiSourceInstanceMixin

from emma_policy.datamodules.relation import Relation


class Task(Enum):
    """Task for the Pretraining instance."""

    mlm = "Masked Language Modelling"
    itm = "Image-Text Matching"
    visual_grounding = "Visual Grounding"
    dense_captioning = "Dense Captioning"
    captioning = "Captioning"
    vqa = "Visual Question Answering"
    relation_detection = "Relationship detection"
    instruction_prediction = "Instruction prediction for trajectory"
    action_execution = "Action execution in embodied environment"
    # TODO: check whether we can put together the entire trajectory
    # goal_prediction = "Goal prediction for an action trajectory"
    vmlm = "Video Masked Language Modelling"
    vtm = "Video-Text Matching"
    fom = "Frame Order Modeling"

    @classmethod
    def get_index(cls, task: Any) -> int:
        """Get task index."""
        return list(cls).index(task)


TASK_TEMPLATES_MAP: Mapping[Task, list[str]] = MappingProxyType(
    {
        Task.mlm: [
            "Denoise: {caption}",
            "Denoise the statement: {caption}",
            "Denoise the description: {caption}",
            "Reconstruct: {caption}",
            "Reconstruct the description: {caption}",
            "Reconstruct the statement: {caption}",
        ],
        Task.itm: [
            "Assess the statement: {statement}",
            "Assess the description: {statement}",
            "Evaluate the statement: {statement}",
            "Evaluate the description: {statement}",
        ],
        Task.visual_grounding: [
            "Find the {caption}",
            "Locate the {caption}",
            "Pick the {caption}",
            "Select the {caption}",
        ],
        Task.dense_captioning: [
            "Caption {region}",
            "Caption object {region}",
            "Describe {region}",
            "Describe object {region}",
        ],
        Task.captioning: [
            "Caption this",
            "Caption the image",
            "Caption this image",
            "Describe this",
            "Describe the image",
            "Describe this image",
        ],
        Task.vqa: [
            "Answer: {question}",
            "Answer the question: {question}",
            "What is the answer to: {question}",
            "What is the answer to the question: {question}",
        ],
        Task.relation_detection: [
            "Explain the relationship between: {subject} and {object}",
            "Explain how {subject} relates to {object}",
            "Describe the relationship between {subject} and {object}",
            "Describe how {subject} relates to {object}",
        ],
        Task.instruction_prediction: [
            "Predict an instruction",
            "Provide an instruction",
        ],
        Task.action_execution: [
            "Act according to the instruction: {instruction}",
            "Execute the instruction: {instruction}",
            "Follow the instruction: {instruction}",
        ],
        Task.vtm: [
            "Assess the statement: {statement}",
            "Assess the description: {statement}",
            "Evaluate the statement: {statement}",
            "Evaluate the description: {statement}",
        ],
        Task.fom: [
            "Order frames based on: {instruction}. ",
            "Order frames based on the instruction: {instruction}. ",
            "Correct the frame order based on: {instruction}",
            "Correct the frame order based on the instruction: {instruction}",
        ],
    }
)


EnabledTasksPerModality = Union[
    dict[str, list[str]],
    dict[MediaType, set[Task]],
]


class EnabledTasksHandler:
    """Collection of methods to process the enabled tasks for a pretraining run."""

    @classmethod
    def get_default_enabled_tasks(cls) -> set[Task]:
        """Get the default enabled tasks.

        In this case, all the tasks are enabled.
        """
        return set(Task)

    @classmethod
    def get_default_enabled_tasks_per_modality(cls) -> dict[MediaType, set[Task]]:
        """Get the default enabled tasks per modality.

        This currently sets all the tasks to be enabled.
        """
        return {media_type: cls.get_default_enabled_tasks() for media_type in MediaType}

    @classmethod
    def process_tasks(cls, tasks: Union[list[str], set[Task]]) -> set[Task]:
        """Ensure tasks are the appropriate enum type."""
        # If one instance is a Task, assume they're all Task enums
        if isinstance(next(iter(tasks)), Task):
            return cast(set[Task], tasks)

        # If it's not correct, assume it's in strings and cast the variable to type correctly.
        tasks = cast(list[str], tasks)
        return cls.process_tasks_from_config(tasks)

    @classmethod
    def process_tasks_per_modality(
        cls, tasks_per_modality: EnabledTasksPerModality
    ) -> dict[MediaType, set[Task]]:
        """Ensure tasks per modality are the appropriate enum type."""
        # If the keys of the outer-most dict are MediaType Enum, then assume it's correct.
        if isinstance(next(iter(tasks_per_modality.keys())), MediaType):
            return cast(dict[MediaType, set[Task]], tasks_per_modality)

        # If it's not correct, assume it's strings and cast the variable to type correctly.
        tasks_per_modality = cast(dict[str, list[str]], tasks_per_modality)
        return cls.process_tasks_per_modality_from_config(tasks_per_modality)

    @classmethod
    def process_tasks_from_config(cls, tasks: list[str]) -> set[Task]:
        """Convert tasks as strings from the config into enums.

        If the list of tasks are empty, just return the empty set.
        """
        if not tasks:
            return set()

        return {Task[task] for task in tasks}

    @classmethod
    def process_tasks_per_modality_from_config(
        cls, tasks_per_modality: dict[str, list[str]]
    ) -> dict[MediaType, set[Task]]:
        """Convert tasks per modality from the config into appropriate enums."""
        return {
            MediaType[media_type]: cls.process_tasks_from_config(enabled_tasks)
            for media_type, enabled_tasks in tasks_per_modality.items()
        }


def extract_task_prefix_strings(templates_map: Mapping[Task, list[str]]) -> Iterator[str]:
    """Generates the string representation associated with each task template."""
    for templates in templates_map.values():
        for template in templates:
            empty_params: dict[Any, str] = {
                name: "" for _, name, _, _ in string.Formatter().parse(template)
            }

            yield template.format(**empty_params)


class PretrainInstance(MultiSourceInstanceMixin):  # type: ignore[misc]
    """Instance for the pretraining datamodule."""

    task: Task
    caption: Optional[Caption]
    qa_pair: Optional[QuestionAnswerPair]
    regions: Optional[list[Region]]
    relations: Optional[list[Relation]]
    trajectory: Optional[ActionTrajectory]
