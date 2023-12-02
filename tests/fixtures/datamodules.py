from pathlib import Path

from pytest_cases import fixture

from emma_policy.datamodules.pretrain_datamodule import EmmaPretrainDataModule


@fixture
def emma_pretrain_datamodule(
    pretrain_db_dir_path: Path,
    model_metadata_path: str,
    enabled_tasks_per_modality: dict[str, list[str]],
) -> EmmaPretrainDataModule:
    """This will create a pretrain db for the given task and make sure that it is available.

    Caching it like this should hopefully reduce the running time/memory usage of the tests.
    """
    dm = EmmaPretrainDataModule(
        pretrain_db_dir_path,
        model_name=model_metadata_path,
        load_valid_data=True,
        enabled_tasks=enabled_tasks_per_modality,
        max_frames=10,
    )
    dm.prepare_data()
    dm.setup()

    return dm


class DataModuleCases:
    def case_pretrain(
        self, emma_pretrain_datamodule: EmmaPretrainDataModule
    ) -> EmmaPretrainDataModule:
        return emma_pretrain_datamodule
