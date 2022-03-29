import itertools
from pathlib import Path

from emma_datasets.datamodels import DatasetSplit, TeachEdhInstance
from emma_datasets.db import DatasetDb
from filelock import FileLock
from pytest_cases import fixture

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem
from emma_policy.datamodules.teach_edh_dataset import TeachEdhDataset
from emma_policy.models.tokenizer_emma import EmmaTokenizer


@fixture
def teach_edh_dataset(
    cached_db_dir_path: Path,
    teach_edh_instances_db: dict[DatasetSplit, Path],
    emma_tokenizer: EmmaTokenizer,
) -> TeachEdhDataset:
    """Merge all the TEACh EDH instances into a single DatasetDB to test all the instances."""
    output_db_path = cached_db_dir_path.joinpath("teach_merged.db")

    with FileLock(cached_db_dir_path.joinpath("teach_edh_dataset.lock")):

        if not output_db_path.exists():
            output_db = DatasetDb(output_db_path, readonly=False)
            teach_split_dbs = itertools.chain.from_iterable(
                [DatasetDb(db_dir) for db_dir in teach_edh_instances_db.values()]
            )

            with output_db:
                data_idx = 0
                for _, _, instance in teach_split_dbs:
                    output_db[(data_idx, f"teach_edh_{data_idx}")] = instance
                    data_idx += 1

    return TeachEdhDataset(dataset_db_path=output_db_path, tokenizer=emma_tokenizer)


def test_dataset_can_get_instances_without_error(teach_edh_dataset: TeachEdhDataset) -> None:
    """Ensure instances can be retrieved without error."""
    total_num_instances = len(teach_edh_dataset.db)

    for idx in range(total_num_instances):
        dataset_item = teach_edh_dataset[idx]
        assert isinstance(dataset_item, EmmaDatasetItem)


def test_dataset_creates_input_text_without_errors(teach_edh_dataset: TeachEdhDataset) -> None:
    """Verify the dataset can create input text without erroring."""
    total_num_instances = len(teach_edh_dataset.db)

    for idx in range(total_num_instances):
        with teach_edh_dataset.db:
            instance_str: str = teach_edh_dataset.db[idx]

        instance = TeachEdhInstance.parse_raw(instance_str)
        input_text = teach_edh_dataset._get_input_text_from_instance(instance)

        assert input_text
        assert isinstance(input_text, str)


def test_dataset_creates_target_text_without_errors(teach_edh_dataset: TeachEdhDataset) -> None:
    """Verify the dataset creates target text without errors."""
    total_num_instances = len(teach_edh_dataset.db)

    for idx in range(total_num_instances):
        with teach_edh_dataset.db:
            instance_str: str = teach_edh_dataset.db[idx]

        instance = TeachEdhInstance.parse_raw(instance_str)
        target_text = teach_edh_dataset._get_target_text_from_instance(instance)

        assert target_text
        assert isinstance(target_text, str)
