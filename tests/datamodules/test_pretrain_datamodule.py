import itertools
from pathlib import Path

from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.pretrain_datamodule import EmmaPretrainDataModule
from emma_policy.datamodules.pretrain_dataset import apply_token_masking
from emma_policy.datamodules.pretrain_instances import (
    PretrainInstance,
    convert_instance_to_pretrain_instances,
)


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


def test_prepare_data_does_not_fail(
    pretrain_db_dir_path: Path, enabled_tasks_per_modality: dict[str, list[str]]
) -> None:
    """Make sure preparing the data works.

    When running the subset, this will verify that there is no issue with reading or writing the
    data to the DatasetDb, which could occur if there are breaking changes made to the underlying
    API.
    """
    dm = EmmaPretrainDataModule(
        pretrain_db_dir_path,
        load_valid_data=True,
        enabled_tasks=enabled_tasks_per_modality,
    )

    dm.prepare_data()
    dm.setup()


def test_dataloader_creates_batches(emma_pretrain_datamodule: EmmaPretrainDataModule) -> None:
    """Ensure that the dataloader can prepare and iterate through all the batches properly."""
    dataloader_iterator = itertools.chain(
        emma_pretrain_datamodule.train_dataloader(), emma_pretrain_datamodule.val_dataloader()
    )

    for batch in dataloader_iterator:
        assert isinstance(batch, EmmaDatasetBatch)


def masked_tokens_check(input_text: str, mask_token: str = "<mask>") -> None:  # noqa: S107
    """Check that the number of tokens does not change and that at least one token is masked."""
    masked_text, _ = apply_token_masking(input_text)
    assert len(masked_text.split()) == len(input_text.split())
    num_masked = sum(1 for token in masked_text.split() if token == mask_token)
    assert num_masked > 0


def test_apply_token_masking(instances_db_path: Path) -> None:
    """Ensure that at least one token is masked for both image-level and region captions."""
    with DatasetDb(instances_db_path) as db:
        for _, _, instance_str in db:
            instance = Instance.parse_raw(instance_str)
            if instance.captions is not None:
                for caption in instance.captions:
                    masked_tokens_check(caption.text)
            if instance.regions is not None:
                for region in instance.regions:
                    masked_tokens_check(region.caption)
