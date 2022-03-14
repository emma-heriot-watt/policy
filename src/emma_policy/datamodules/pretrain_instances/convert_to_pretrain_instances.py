import itertools
from typing import Callable, Iterator, Optional

from emma_datasets.datamodels import Instance, MediaType

from emma_policy.datamodules.pretrain_instances.datamodels import (
    EnabledTasksHandler,
    PretrainInstance,
    Task,
)


PretrainInstanceCreatorFuncType = Callable[["PretrainInstanceCreator"], Iterator[PretrainInstance]]


def image_task_check(func: PretrainInstanceCreatorFuncType) -> PretrainInstanceCreatorFuncType:
    """Makes sure that the current instance belongs to the image-based dataset."""

    def wrapper(creator: "PretrainInstanceCreator") -> Iterator[PretrainInstance]:
        if creator.instance.modality != MediaType.image:
            return iter([])
        return func(creator)

    return wrapper


def video_task_check(func: PretrainInstanceCreatorFuncType) -> PretrainInstanceCreatorFuncType:
    """Makes sure that the current instance belongs to the video-based dataset."""

    def wrapper(creator: "PretrainInstanceCreator") -> Iterator[PretrainInstance]:
        if creator.instance.modality != MediaType.video:
            return iter([])
        return func(creator)

    return wrapper


class PretrainInstanceCreator:
    """Create PretrainInstances from Instances.

    Each task has a property which returns an iterator of `PretrainInstance`, which can be of any
    length.

    For each task-related property, an empty list is returned if the instance does not support the
    task. This will then be ignored when iterating over all the properties.
    """

    def __init__(self, instance: Instance, enabled_tasks: Optional[set[Task]] = None) -> None:
        self.instance = instance

        self.enabled_tasks = (
            enabled_tasks
            if enabled_tasks is not None
            else EnabledTasksHandler.get_default_enabled_tasks()
        )

        self.instance_task_map: dict[Task, Iterator[PretrainInstance]] = {
            Task.mlm: self.mlm,
            Task.itm: self.itm,
            Task.visual_grounding: self.visual_grounding,
            Task.dense_captioning: self.dense_captioning,
            Task.captioning: self.captioning,
            Task.vqa: self.vqa,
            Task.instruction_prediction: self.instruction_prediction,
            Task.action_execution: self.action_execution,
            Task.vtm: self.vtm,
        }

        if list(self.instance_task_map.keys()) != list(Task):
            raise AssertionError(
                "Not all pretrain tasks are accounted for by the pretraining instance processor."
            )

    def get_all_pretrain_instances(self) -> Iterator[PretrainInstance]:
        """Get all the pretrain instances for the current instance."""
        yield from itertools.chain.from_iterable(self.instance_task_map.values())

    def __getitem__(self, task: Task) -> Iterator[PretrainInstance]:
        """Get the pretraining instances for the given task."""
        return self.instance_task_map[task]

    @property  # type: ignore[misc]
    @image_task_check
    def mlm(self) -> Iterator[PretrainInstance]:
        """Get pretrain instances for the MLM task."""
        if self.instance.caption is None or Task.mlm not in self.enabled_tasks:
            return []

        yield PretrainInstance(
            caption=self.instance.caption, dataset=self.instance.dataset, task=Task.mlm
        )

    @property  # type: ignore[misc]
    @image_task_check
    def itm(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instances for the ITM task."""
        if self.instance.caption is None or Task.itm not in self.enabled_tasks:
            return []

        yield PretrainInstance(
            caption=self.instance.caption, dataset=self.instance.dataset, task=Task.itm
        )

    @property  # type: ignore[misc]
    @image_task_check
    def visual_grounding(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instances for the visual grounding task."""
        if self.instance.regions is None or Task.visual_grounding not in self.enabled_tasks:
            return []

        yield PretrainInstance(
            regions=self.instance.regions,
            dataset=self.instance.dataset,
            task=Task.visual_grounding,
        )

    @property  # type: ignore[misc]
    @image_task_check
    def dense_captioning(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instances for the dense captioning task."""
        if self.instance.regions is None or Task.dense_captioning not in self.enabled_tasks:
            return []

        yield PretrainInstance(
            regions=self.instance.regions,
            dataset=self.instance.dataset,
            task=Task.dense_captioning,
        )

    @property  # type: ignore[misc]
    @image_task_check
    def captioning(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instances for the captioning task."""
        if self.instance.caption is None or Task.captioning not in self.enabled_tasks:
            return []

        yield PretrainInstance(
            caption=self.instance.caption, dataset=self.instance.dataset, task=Task.captioning
        )

    @property  # type: ignore[misc]
    @image_task_check
    def vqa(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instances for the VQA task."""
        if self.instance.qa is None or Task.vqa not in self.enabled_tasks:
            return []

        yield PretrainInstance(
            qa=self.instance.qa,
            dataset=self.instance.dataset,
            task=Task.vqa,
        )

    @property  # type: ignore[misc]
    @video_task_check
    def instruction_prediction(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instance for the instruction prediction for a given subgoal."""
        skip_instance = (
            self.instance.trajectory is None
            or self.instance.caption is None
            or Task.instruction_prediction not in self.enabled_tasks
        )
        if skip_instance:
            return []

        yield PretrainInstance(
            caption=self.instance.caption,
            trajectory=self.instance.trajectory,
            dataset=self.instance.dataset,
            task=Task.instruction_prediction,
        )

    @property  # type: ignore[misc]
    @video_task_check
    def action_execution(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instance for the action execution task given a subgoal instruction."""
        skip_instance = (
            self.instance.trajectory is None
            or self.instance.caption is None
            or Task.action_execution not in self.enabled_tasks
        )
        if skip_instance:
            return []

        yield PretrainInstance(
            caption=self.instance.caption,
            trajectory=self.instance.trajectory,
            dataset=self.instance.dataset,
            task=Task.action_execution,
        )

    @property  # type: ignore[misc]
    @video_task_check
    def vtm(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instance for the video-text matching task given a subgoal."""
        if self.instance.caption is None or Task.vtm not in self.enabled_tasks:
            return []

        yield PretrainInstance(
            caption=self.instance.caption, dataset=self.instance.dataset, task=Task.vtm
        )


def convert_instance_to_pretrain_instances(
    instance: Instance, enabled_tasks: Optional[set[Task]] = None
) -> Iterator[PretrainInstance]:
    """Convert an instance to all possible pretrain instances."""
    pretrain_instance_creator = PretrainInstanceCreator(instance, enabled_tasks=enabled_tasks)
    yield from pretrain_instance_creator.get_all_pretrain_instances()
