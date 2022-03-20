from pathlib import Path
from typing import Any, Literal, Optional

from pytest_cases import fixture, parametrize

from emma_policy.datamodules.pretrain_datamodule import EmmaPretrainDataModule
from emma_policy.datamodules.teach_edh_datamodule import TeachEdhDataModule


@fixture
def emma_pretrain_datamodule(
    request: Any,
    cached_db_dir_path: Path,
    instances_db_path: Path,
    model_metadata_path: str,
    enabled_tasks_per_modality: dict[str, list[str]],
) -> EmmaPretrainDataModule:
    """This will create a pretrain db for the given task and make sure that it is available.

    Caching it like this should hopefully reduce the running time/memory usage of the tests.
    """
    db_suffix = request.node.callspec.id

    dm = EmmaPretrainDataModule(
        cached_db_dir_path.joinpath(f"pretrain_train_{db_suffix}.db"),
        cached_db_dir_path.joinpath(f"pretrain_valid_{db_suffix}.db"),
        instances_db_path,
        model_name=model_metadata_path,
        load_valid_data=True,
        enabled_tasks=enabled_tasks_per_modality,
    )
    dm.prepare_data()
    dm.setup()

    return dm


@fixture
@parametrize("valid_data_split", [None, "seen", "unseen", "both"])
def teach_edh_datamodule(
    teach_edh_instances_db: Path, valid_data_split: Optional[Literal["seen", "unseen", "both"]]
) -> TeachEdhDataModule:
    datamodule = TeachEdhDataModule(
        teach_edh_instances_db,
        teach_edh_instances_db,
        teach_edh_instances_db,
        load_valid_data_split=valid_data_split,
    )
    datamodule.prepare_data()
    datamodule.setup()

    return datamodule
