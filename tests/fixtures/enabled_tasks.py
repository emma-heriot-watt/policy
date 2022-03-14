import pytest
from pytest_cases import param_fixture

from emma_policy.datamodules.pretrain_instances.datamodels import Task


all_tasks_list = [
    task.name for task in Task if task not in {Task.instruction_prediction, Task.vtm}
]

enabled_tasks_list = param_fixture(
    "enabled_tasks_list",
    [
        pytest.param(["mlm"], id="single_task"),
        pytest.param(["mlm", "itm", "visual_grounding"], id="multiple_tasks"),
        pytest.param(["mlm", "mlm"], id="repeated_tasks"),
        pytest.param(
            ["mlm", "mtm"], marks=pytest.mark.xfail(raises=KeyError), id="misspelled_task"
        ),
        pytest.param(all_tasks_list, id="all_tasks"),
    ],
)


enabled_tasks_per_modality = param_fixture(
    "enabled_tasks_per_modality",
    [
        pytest.param(
            {"image": ["mlm"], "video": []},
            id="single_task",
        ),
        pytest.param(
            {"image": ["mlm", "itm", "captioning"], "video": ["action_execution"]},
            id="multiple_tasks",
        ),
        pytest.param(
            {"image": ["mlm", "mlm"], "video": []},
            id="repeated_tasks",
        ),
        pytest.param(
            {"image": ["mlm", "mtm"], "video": []},
            marks=pytest.mark.xfail(raises=KeyError),
            id="misspelled_task",
        ),
        pytest.param(
            {"img": ["mlm"], "video": []},
            marks=pytest.mark.xfail(raises=KeyError),
            id="misspelled_modality",
        ),
        pytest.param(
            {"image": ["mlm"], "video": []},
            id="one_empty_modality",
        ),
        pytest.param(
            {"image": [], "video": []},
            marks=pytest.mark.skip("no logic present to handle this case"),
            id="all_empty_modality",
        ),
        pytest.param(
            {"image": all_tasks_list, "video": all_tasks_list},
            id="all_tasks",
        ),
    ],
)
