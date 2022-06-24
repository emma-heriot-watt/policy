import itertools
from typing import Callable, Iterator, Optional

from emma_datasets.datamodels import Caption, Instance, MediaType, Region

from emma_policy.datamodules.pretrain_instances.datamodels import (
    EnabledTasksHandler,
    PretrainInstance,
    Task,
)
from emma_policy.datamodules.relation import Relation


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
            Task.relation_detection: self.relation_detection,
            Task.instruction_prediction: self.instruction_prediction,
            Task.action_execution: self.action_execution,
            Task.goal_prediction: self.goal_prediction,
            Task.vmlm: self.vmlm,
            Task.vtm: self.vtm,
            Task.fom: self.fom,
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
        if Task.mlm not in self.enabled_tasks:
            return []

        all_captions: list[PretrainInstance] = []

        if self.instance.captions:
            all_captions.extend(
                PretrainInstance(caption=caption, dataset=self.instance.dataset, task=Task.mlm)
                for caption in self.instance.captions
            )

        if self.instance.regions:
            # due to overlapping regions, we make sure that they are unique
            unique_captions = set()

            for region in self.instance.regions:
                if region.caption not in unique_captions:
                    unique_captions.add(region.caption)
                    all_captions.append(
                        PretrainInstance(
                            caption=Caption(text=region.caption),
                            dataset=self.instance.dataset,
                            task=Task.mlm,
                        )
                    )

        yield from all_captions

    @property  # type: ignore[misc]
    @image_task_check
    def itm(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instances for the ITM task."""
        if not self.instance.captions or Task.itm not in self.enabled_tasks:
            return []

        yield from (
            PretrainInstance(caption=caption, dataset=self.instance.dataset, task=Task.itm)
            for caption in self.instance.captions
        )

    @property  # type: ignore[misc]
    @image_task_check
    def visual_grounding(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instances for the visual grounding task."""
        if not self.instance.regions or Task.visual_grounding not in self.enabled_tasks:
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
        if not self.instance.regions or Task.dense_captioning not in self.enabled_tasks:
            return []

        yield PretrainInstance(
            regions=self.instance.regions,
            dataset=self.instance.dataset,
            task=Task.dense_captioning,
        )

    @property  # type: ignore[misc]
    @image_task_check
    def relation_detection(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instances for the relation detection task."""
        if self.instance.scene_graph is None or not self.instance.scene_graph.objects.keys():
            return []
        if Task.relation_detection not in self.enabled_tasks:
            return []

        flatten_relations = []
        for _id, gqa_obj in self.instance.scene_graph.objects.items():
            for gqa_rel in gqa_obj.relations:
                subject = Region(
                    caption=gqa_obj.name,
                    bbox=[
                        gqa_obj.x,
                        gqa_obj.y,
                        gqa_obj.w,
                        gqa_obj.h,
                    ],
                )
                rel_object = self.instance.scene_graph.objects[gqa_rel.object]
                object_ = Region(
                    caption=rel_object.name,
                    bbox=[
                        rel_object.x,
                        rel_object.y,
                        rel_object.w,
                        rel_object.h,
                    ],
                )
                flatten_relations.append(
                    Relation(
                        subject_attr=gqa_obj.attributes,
                        subject=subject,
                        predicate=gqa_rel.name,
                        object=object_,
                        object_attr=rel_object.attributes,
                    )
                )

        yield PretrainInstance(
            relations=flatten_relations,
            dataset=self.instance.dataset,
            task=Task.relation_detection,
        )

    @property  # type: ignore[misc]
    @image_task_check
    def captioning(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instances for the captioning task."""
        if not self.instance.captions or Task.captioning not in self.enabled_tasks:
            return []

        yield from (
            PretrainInstance(caption=caption, dataset=self.instance.dataset, task=Task.captioning)
            for caption in self.instance.captions
        )

    @property  # type: ignore[misc]
    @image_task_check
    def vqa(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instances for the VQA task."""
        if not self.instance.qa_pairs or Task.vqa not in self.enabled_tasks:
            return []

        yield from (
            PretrainInstance(
                qa_pair=qa_pair,
                dataset=self.instance.dataset,
                task=Task.vqa,
            )
            for qa_pair in self.instance.qa_pairs
        )

    @property  # type: ignore[misc]
    @video_task_check
    def instruction_prediction(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instance for the instruction prediction for a given subgoal."""
        skip_instance = (
            not self.instance.captions or Task.instruction_prediction not in self.enabled_tasks
        )
        if skip_instance:
            return []

        yield from (
            PretrainInstance(
                caption=caption,
                trajectory=self.instance.trajectory,
                dataset=self.instance.dataset,
                task=Task.instruction_prediction,
            )
            for caption in self.instance.captions
        )

    @property  # type: ignore[misc]
    @video_task_check
    def goal_prediction(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instance for the goal prediction for a given trajectory."""
        skip_instance = (
            not self.instance.is_full_trajectory
            or Task.goal_prediction not in self.enabled_tasks
            or not self.instance.task_description
        )
        if skip_instance:
            return []

        yield from (
            PretrainInstance(
                task_description=task_description,
                trajectory=self.instance.trajectory,
                dataset=self.instance.dataset,
                task=Task.goal_prediction,
            )
            for task_description in self.instance.task_description
        )

    @property  # type: ignore[misc]
    @video_task_check
    def action_execution(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instance for the action execution task given a subgoal instruction."""
        skip_instance = (
            self.instance.trajectory is None
            or not self.instance.captions
            or Task.action_execution not in self.enabled_tasks
        )
        if skip_instance:
            return []

        yield from (
            PretrainInstance(
                caption=caption,
                trajectory=self.instance.trajectory,
                dataset=self.instance.dataset,
                task=Task.action_execution,
            )
            for caption in self.instance.captions
        )

    @property  # type: ignore[misc]
    @video_task_check
    def vtm(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instance for the video-text matching task given a subgoal."""
        skip_instance = (
            not self.instance.captions
            or Task.vtm not in self.enabled_tasks
            or self.instance.is_full_trajectory  # Do not apply VTM to trajectory data
        )
        if skip_instance:
            return []

        yield from (
            PretrainInstance(
                caption=caption,
                trajectory=self.instance.trajectory,
                dataset=self.instance.dataset,
                task=Task.vtm,
            )
            for caption in self.instance.captions
        )

    @property  # type: ignore[misc]
    @video_task_check
    def fom(self) -> Iterator[PretrainInstance]:
        """Get the pretrain instance for the feature order modeling task given a subgoal."""
        skip_instance = not self.instance.captions or Task.fom not in self.enabled_tasks
        if skip_instance:
            return []

        yield from (
            PretrainInstance(
                caption=caption,
                trajectory=self.instance.trajectory,
                dataset=self.instance.dataset,
                task=Task.fom,
            )
            for caption in self.instance.captions
        )

    @property  # type: ignore[misc]
    @video_task_check
    def vmlm(self) -> Iterator[PretrainInstance]:
        """Get pretrain instances for the video MLM task."""
        if not self.instance.captions or Task.vmlm not in self.enabled_tasks:
            return []

        yield from (
            PretrainInstance(
                caption=caption,
                trajectory=self.instance.trajectory,
                dataset=self.instance.dataset,
                task=Task.vmlm,
            )
            for caption in self.instance.captions
        )


def convert_instance_to_pretrain_instances(
    instance: Instance, enabled_tasks: Optional[set[Task]] = None
) -> Iterator[PretrainInstance]:
    """Convert an instance to all possible pretrain instances."""
    pretrain_instance_creator = PretrainInstanceCreator(instance, enabled_tasks=enabled_tasks)
    yield from pretrain_instance_creator.get_all_pretrain_instances()
