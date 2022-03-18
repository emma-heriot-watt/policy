from pathlib import Path

from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb
from pytest_cases import parametrize

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.pretrain_datamodule import EmmaPretrainDataModule
from emma_policy.datamodules.pretrain_dataset import split_action_name
from emma_policy.datamodules.pretrain_instances import (
    PretrainInstance,
    convert_instance_to_pretrain_instances,
    load_ref_coco_images,
)


def test_load_coco_valid_ids() -> None:
    coco_splits_path = Path("storage/constants/mscoco_resplit_val.json")
    valid_ids = load_ref_coco_images(coco_splits_path)

    assert valid_ids


def test_prepare_data_runs_without_failing(
    tmp_path: Path, instances_db_path: Path, enabled_tasks_per_modality: dict[str, list[str]]
) -> None:
    """Make sure preparing the data works.

    When running the subset, this will verify that there is no issue with reading or writing the
    data to the DatasetDb, which could occur if there are breaking changes made to the underlying
    API.
    """
    dm = EmmaPretrainDataModule(
        tmp_path.joinpath("pretrain_train.db"),
        tmp_path.joinpath("pretrain_valid.db"),
        instances_db_path,
        force_prepare_data=True,
        enabled_tasks=enabled_tasks_per_modality,
    )

    dm.prepare_data()


def test_instances_convert_to_pretrain_instances(instances_db_path: Path) -> None:
    """Verify instances are converted to pretrain instances without failing.

    The most important difference is that each pretrain instance declares its task. Pydantic should
    not allow this to happen anyway.
    """
    with DatasetDb(instances_db_path) as db:
        for _, _, instance_str in db:
            instance = Instance.parse_raw(instance_str)

            pretrain_instances = convert_instance_to_pretrain_instances(instance)

            for pretrain_instance in pretrain_instances:
                assert isinstance(pretrain_instance, PretrainInstance)


def test_dataloader_creates_batches_properly(
    tmp_path: Path, instances_db_path: Path, enabled_tasks_per_modality: dict[str, list[str]]
) -> None:
    """Ensure that the dataloader can prepare and iterate through all the batches properly."""
    dm = EmmaPretrainDataModule(
        tmp_path.joinpath("pretrain_train.db"),
        tmp_path.joinpath("pretrain_valid.db"),
        instances_db_path,
        force_prepare_data=True,
        enabled_tasks=enabled_tasks_per_modality,
    )
    dm.prepare_data()
    dm.setup()

    for batch in iter(dm.train_dataloader()):
        assert isinstance(batch, EmmaDatasetBatch)


@parametrize(
    "action_name, action_text",
    [
        ("Pickup", "pick up"),
        ("PickupObject", "pick up"),
        ("Stop", "stop"),
        ("Move to", "move to"),
        ("Forward", "forward"),
        ("Turn Left", "turn left"),
        ("Look Up", "look up"),
        ("ToggleOn", "toggle on"),
        ("ToggleOff", "toggle off"),
        ("BehindAboveOn", "behind above on"),
        ("OpenProgressCheck", "open progress check"),
    ],
)
def test_convert_action_name_to_consistent_form(action_name: str, action_text: str) -> None:
    assert split_action_name(action_name) == action_text.split(" ")
