from pathlib import Path

from pytest_cases import fixture

from emma_policy.datamodules.pretrain_datamodule import EmmaPretrainDataModule


@fixture
def emma_pretrain_datamodule(
    tmp_path: Path,
    instances_db_path: Path,
    model_metadata_path: str,
    enabled_tasks_per_modality: dict[str, list[str]],
) -> EmmaPretrainDataModule:
    dm = EmmaPretrainDataModule(
        tmp_path.joinpath("pretrain_train.db"),
        tmp_path.joinpath("pretrain_valid.db"),
        instances_db_path,
        model_name=model_metadata_path,
        force_prepare_data=True,
        load_valid_data=True,
        enabled_tasks=enabled_tasks_per_modality,
    )
    dm.prepare_data()
    dm.setup()

    return dm
