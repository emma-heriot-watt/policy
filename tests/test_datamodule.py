from pathlib import Path

import pytest
from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb
from pytest_cases import fixture_ref, parametrize

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.pretrain_datamodule import EmmaPretrainDataModule
from emma_policy.datamodules.pretrain_instances import (
    PretrainInstance,
    convert_instance_to_pretrain_instances,
    load_ref_coco_images,
)


def test_load_coco_valid_ids() -> None:
    coco_splits_path = Path("storage/constants/mscoco_resplit_val.json")
    valid_ids = load_ref_coco_images(coco_splits_path)

    assert valid_ids


@parametrize(
    "instances_db_path",
    [
        pytest.param(Path("storage/fixtures/instances.db"), marks=pytest.mark.slow, id="full"),
        pytest.param(fixture_ref("tiny_instances_db_path"), id="subset"),
    ],
)
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


def test_instances_convert_to_pretrain_instances() -> None:
    """Verify instances are converted to pretrain instances without failing.

    The most important difference is that each pretrain instance declares its task. Pydantic should
    not allow this to happen anyway.
    """
    with DatasetDb("storage/fixtures/instances.db") as db:
        for _, _, instance_str in db:
            instance = Instance.parse_raw(instance_str)

            pretrain_instances = convert_instance_to_pretrain_instances(instance)

            for pretrain_instance in pretrain_instances:
                assert isinstance(pretrain_instance, PretrainInstance)


@parametrize(
    "instances_db_path",
    [
        pytest.param(Path("storage/fixtures/instances.db"), marks=pytest.mark.slow, id="full"),
        pytest.param(fixture_ref("tiny_instances_db_path"), id="subset"),
    ],
)
def test_dataloader_iterator(
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
    dm.setup()

    # once the data are ready, we inspect the train dataloader
    train_loader = dm.train_dataloader()

    assert train_loader, "No training data loader available"


def test_dataloader_batch(
    tmp_path: Path,
    instances_tiny_batch_db_path: Path,
    enabled_tasks_per_modality: dict[str, list[str]],
) -> None:
    """Make sure preparing the data works.

    When running the subset, this will verify that there is no issue with reading or writing the
    data to the DatasetDb, which could occur if there are breaking changes made to the underlying
    API.
    """
    dm = EmmaPretrainDataModule(
        tmp_path.joinpath("pretrain_train.db"),
        tmp_path.joinpath("pretrain_valid.db"),
        instances_tiny_batch_db_path,
        force_prepare_data=True,
        batch_size=2,
        load_valid_data=False,
        enabled_tasks=enabled_tasks_per_modality,
    )
    dm.prepare_data()
    dm.setup()

    # once the data are ready, we inspect the train dataloader
    train_loader = dm.train_dataloader()

    assert train_loader, "No training data loader available"

    assert isinstance(next(iter(train_loader)), EmmaDatasetBatch)
