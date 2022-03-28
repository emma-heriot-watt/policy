from pathlib import Path
from typing import Any, Literal

from emma_datasets.datamodels import DatasetSplit
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
@parametrize("valid_data_split", ["seen", "unseen", "both"])
def teach_edh_datamodule(
    teach_edh_instances_db: dict[DatasetSplit, Path],
    valid_data_split: Literal["seen", "unseen", "both"],
) -> TeachEdhDataModule:
    datamodule = TeachEdhDataModule(
        teach_edh_train_db_file=teach_edh_instances_db[DatasetSplit.train],
        teach_edh_valid_seen_db_file=teach_edh_instances_db[DatasetSplit.valid_seen],
        teach_edh_valid_unseen_db_file=teach_edh_instances_db[DatasetSplit.valid_unseen],
        load_valid_data_split=valid_data_split,
    )
    datamodule.prepare_data()
    datamodule.setup()

    return datamodule


class TeachEdhDataModuleCases:
    def case_valid_seen(
        self, teach_edh_instances_db: dict[DatasetSplit, Path]
    ) -> TeachEdhDataModule:
        datamodule = TeachEdhDataModule(
            teach_edh_train_db_file=teach_edh_instances_db[DatasetSplit.train],
            teach_edh_valid_seen_db_file=teach_edh_instances_db[DatasetSplit.valid_seen],
            teach_edh_valid_unseen_db_file=teach_edh_instances_db[DatasetSplit.valid_unseen],
            load_valid_data_split="seen",
        )
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule

    def case_valid_unseen(
        self, teach_edh_instances_db: dict[DatasetSplit, Path]
    ) -> TeachEdhDataModule:
        datamodule = TeachEdhDataModule(
            teach_edh_train_db_file=teach_edh_instances_db[DatasetSplit.train],
            teach_edh_valid_seen_db_file=teach_edh_instances_db[DatasetSplit.valid_seen],
            teach_edh_valid_unseen_db_file=teach_edh_instances_db[DatasetSplit.valid_unseen],
            load_valid_data_split="unseen",
        )
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule

    def case_valid_seen_and_unseen(
        self, teach_edh_instances_db: dict[DatasetSplit, Path]
    ) -> TeachEdhDataModule:
        datamodule = TeachEdhDataModule(
            teach_edh_train_db_file=teach_edh_instances_db[DatasetSplit.train],
            teach_edh_valid_seen_db_file=teach_edh_instances_db[DatasetSplit.valid_seen],
            teach_edh_valid_unseen_db_file=teach_edh_instances_db[DatasetSplit.valid_unseen],
            load_valid_data_split="both",
        )
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule


class DataModuleCases:
    def case_pretrain(
        self, emma_pretrain_datamodule: EmmaPretrainDataModule
    ) -> EmmaPretrainDataModule:
        return emma_pretrain_datamodule

    def case_teach_edh_datamodule(
        self, teach_edh_datamodule: TeachEdhDataModule
    ) -> TeachEdhDataModule:
        return teach_edh_datamodule
