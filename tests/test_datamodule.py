from pathlib import Path

import pytest
from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb
from pytest_cases import fixture_ref, parametrize

from emma_policy.datamodules.pretrain_datamodule import EmmaPretrainDataModule
from emma_policy.datamodules.pretrain_instances import (
    PretrainInstance,
    convert_instance_to_pretrain_instances,
)


@parametrize(
    "instances_db_path",
    [
        pytest.param(Path("storage/fixtures/instances.db"), marks=pytest.mark.slow, id="full"),
        pytest.param(fixture_ref("tiny_instances_db_path"), id="subset"),
    ],
)
def test_prepare_data_runs_without_failing(tmp_path: Path, instances_db_path: Path) -> None:
    """Make sure preparing the data works.

    When running the subset, this will verify that there is no issue with reading or writing the
    data to the DatasetDb, which could occur if there are breaking changes made to the underlying
    API.
    """
    dm = EmmaPretrainDataModule(
        tmp_path.joinpath("pretrain.db"), instances_db_path, force_prepare_data=True
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
