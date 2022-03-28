from pytest_cases import parametrize_with_cases
from torch.utils.data import ConcatDataset

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.teach_edh_datamodule import TeachEdhDataModule
from tests.fixtures.datamodules import TeachEdhDataModuleCases


def test_dataloader_creates_train_batches(teach_edh_datamodule: TeachEdhDataModule) -> None:
    # Ensure that the train dataloader is making batches
    for batch in iter(teach_edh_datamodule.train_dataloader()):
        assert isinstance(batch, EmmaDatasetBatch)


@parametrize_with_cases("teach_edh_datamodule", cases=TeachEdhDataModuleCases, glob="valid_seen")
def test_dataloader_creates_valid_seen_batches(teach_edh_datamodule: TeachEdhDataModule) -> None:
    valid_dataloader = teach_edh_datamodule.val_dataloader()

    # Ensure the valid dataloder is using the valid seen dataset
    assert valid_dataloader.dataset == teach_edh_datamodule._valid_seen_dataset

    # Ensure that the valid dataloader is making batches
    for batch in teach_edh_datamodule.val_dataloader():
        assert isinstance(batch, EmmaDatasetBatch)


@parametrize_with_cases("teach_edh_datamodule", cases=TeachEdhDataModuleCases, glob="valid_unseen")
def test_dataloader_creates_valid_unseen_batches(teach_edh_datamodule: TeachEdhDataModule) -> None:
    valid_dataloader = teach_edh_datamodule.val_dataloader()

    # Ensure the valid dataloder is using the valid unseen dataset
    assert valid_dataloader.dataset == teach_edh_datamodule._valid_unseen_dataset

    # Ensure that the valid dataloader is making batches
    for batch in teach_edh_datamodule.val_dataloader():
        assert isinstance(batch, EmmaDatasetBatch)


@parametrize_with_cases(
    "teach_edh_datamodule", cases=TeachEdhDataModuleCases, glob="valid_seen_and_unseen"
)
def test_dataloader_uses_both_seen_and_unseen_valid_instances(
    teach_edh_datamodule: TeachEdhDataModule,
) -> None:
    valid_dataloader = teach_edh_datamodule.val_dataloader()

    # Ensure the dataset given to the dataloder is the ConcatDataset
    assert isinstance(valid_dataloader.dataset, ConcatDataset)

    # Ensure that both the valid seen and valid unseen datasets are in the ConcatDataset
    for dataset in valid_dataloader.dataset.datasets:
        assert dataset in {
            teach_edh_datamodule._valid_seen_dataset,
            teach_edh_datamodule._valid_unseen_dataset,
        }
        assert dataset != teach_edh_datamodule._train_dataset

    # Ensure that the valid dataloader is making batches
    for batch in teach_edh_datamodule.val_dataloader():
        assert isinstance(batch, EmmaDatasetBatch)
