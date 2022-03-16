from pathlib import Path

from torch.utils.data import ConcatDataset

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.teach_edh_datamodule import TeachEdhDataModule


def test_dataloader_creates_train_batches(teach_edh_instances_db: Path) -> None:
    datamodule = TeachEdhDataModule(
        teach_edh_instances_db,
        teach_edh_instances_db,
        teach_edh_instances_db,
    )
    datamodule.prepare_data()
    datamodule.setup()

    # Ensure that the train dataloader is making batches
    for batch in iter(datamodule.train_dataloader()):
        assert isinstance(batch, EmmaDatasetBatch)


def test_dataloader_creates_valid_seen_batches(teach_edh_instances_db: Path) -> None:
    datamodule = TeachEdhDataModule(
        teach_edh_instances_db,
        teach_edh_instances_db,
        teach_edh_instances_db,
        load_valid_data_split="seen",
    )
    datamodule.prepare_data()
    datamodule.setup()

    valid_dataloader = datamodule.val_dataloader()

    # Ensure the valid dataloder is using the valid seen dataset
    assert valid_dataloader.dataset == datamodule._valid_seen_dataset

    # Ensure that the valid dataloader is making batches
    for batch in iter(datamodule.val_dataloader()):
        assert isinstance(batch, EmmaDatasetBatch)


def test_dataloader_creates_valid_unseen_batches(teach_edh_instances_db: Path) -> None:
    datamodule = TeachEdhDataModule(
        teach_edh_instances_db,
        teach_edh_instances_db,
        teach_edh_instances_db,
        load_valid_data_split="unseen",
    )
    datamodule.prepare_data()
    datamodule.setup()

    valid_dataloader = datamodule.val_dataloader()

    # Ensure the valid dataloder is using the valid unseen dataset
    assert valid_dataloader.dataset == datamodule._valid_unseen_dataset

    # Ensure that the valid dataloader is making batches
    for batch in iter(datamodule.val_dataloader()):
        assert isinstance(batch, EmmaDatasetBatch)


def test_dataloader_uses_both_seen_and_unseen_valid_instances(
    teach_edh_instances_db: Path,
) -> None:
    datamodule = TeachEdhDataModule(
        teach_edh_instances_db,
        teach_edh_instances_db,
        teach_edh_instances_db,
        load_valid_data_split="both",
    )
    datamodule.prepare_data()
    datamodule.setup()

    valid_dataloader = datamodule.val_dataloader()

    # Ensure the dataset given to the dataloder is the ConcatDataset
    assert isinstance(valid_dataloader.dataset, ConcatDataset)

    # Ensure that both the valid seen and valid unseen datasets are in the ConcatDataset
    for dataset in valid_dataloader.dataset.datasets:
        assert dataset in {datamodule._valid_seen_dataset, datamodule._valid_unseen_dataset}
        assert dataset != datamodule._train_dataset

    # Ensure that the valid dataloader is making batches
    for batch in iter(datamodule.val_dataloader()):
        assert isinstance(batch, EmmaDatasetBatch)
