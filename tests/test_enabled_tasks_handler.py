from emma_datasets.datamodels import MediaType

from emma_policy.datamodules.pretrain_instances.datamodels import EnabledTasksHandler, Task


def test_all_tasks_enabled_by_default() -> None:
    default_tasks = EnabledTasksHandler.get_default_enabled_tasks()

    for task in Task:
        assert task in default_tasks


def test_all_tasks_enabled_by_default_for_all_modality() -> None:
    default_tasks_per_modality = EnabledTasksHandler.get_default_enabled_tasks_per_modality()

    for modality, default_tasks in default_tasks_per_modality.items():
        assert isinstance(modality, MediaType)

        for task in Task:
            assert task in default_tasks


def test_can_process_tasks_from_string_list(enabled_tasks_list: list[str]) -> None:
    processed_tasks = EnabledTasksHandler.process_tasks_from_config(enabled_tasks_list)

    for task in processed_tasks:
        assert isinstance(task, Task)


def test_can_process_tasks_per_modality_from_strings(
    enabled_tasks_per_modality: dict[str, list[str]]
) -> None:
    processed_tasks = EnabledTasksHandler.process_tasks_per_modality_from_config(
        enabled_tasks_per_modality
    )

    for modality, tasks in processed_tasks.items():
        assert isinstance(modality, MediaType)

        if tasks:
            for task in tasks:
                assert isinstance(task, Task)
