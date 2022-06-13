import itertools
from pathlib import Path

import torch
from emma_datasets.datamodels import DatasetSplit
from emma_datasets.datamodels.datasets import TeachEdhInstance
from emma_datasets.db import DatasetDb
from filelock import FileLock
from pytest_cases import fixture, parametrize

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
        visual_features, _, _ = teach_edh_dataset._prepare_visual_input(instance)
        input_text = teach_edh_dataset._get_input_text_from_instance(instance, visual_features)

        assert input_text
        assert isinstance(input_text, str)


def test_dataset_creates_target_text_without_errors(teach_edh_dataset: TeachEdhDataset) -> None:
    """Verify the dataset creates target text without errors."""
    total_num_instances = len(teach_edh_dataset.db)

    for idx in range(total_num_instances):
        with teach_edh_dataset.db:
            instance_str: str = teach_edh_dataset.db[idx]

        instance = TeachEdhInstance.parse_raw(instance_str)
        visual_features, _, _ = teach_edh_dataset._prepare_visual_input(instance)
        target_text = teach_edh_dataset._get_target_text_from_instance(instance, visual_features)

        assert target_text
        assert isinstance(target_text, str)


@parametrize("unknown_visual_token_threshold", [0.5])
def test_parsed_visual_tokens_are_not_all_unknown(
    teach_edh_dataset: TeachEdhDataset,
    emma_tokenizer: EmmaTokenizer,
    unknown_visual_token_threshold: float,
) -> None:
    total_num_instances = len(teach_edh_dataset.db)

    for idx in range(total_num_instances):
        with teach_edh_dataset.db:
            instance_str: str = teach_edh_dataset.db[idx]

        instance = TeachEdhInstance.parse_raw(instance_str)
        visual_features, _, _ = teach_edh_dataset._prepare_visual_input(instance)

        # Checking the unknowns in the action history
        input_text = teach_edh_dataset._get_input_text_from_instance(instance, visual_features)
        target_text = teach_edh_dataset._get_target_text_from_instance(instance, visual_features)

        all_interaction_actions = list(
            filter(
                lambda action: action.obj_interaction_action == 1,
                itertools.chain(instance.driver_action_history, instance.driver_actions_future),
            )
        )

        # Get the maximum number of visual tokens
        max_visual_token_count = len(all_interaction_actions)

        parsed_unk_token_count = input_text.count(emma_tokenizer.unk_token) + target_text.count(
            emma_tokenizer.unk_token
        )

        if parsed_unk_token_count > max_visual_token_count * unknown_visual_token_threshold:
            raise AssertionError("The number of unknowns in the `input_text` is too high.")


@parametrize(
    "target_tokens,expected_target_tokens",
    [
        (
            torch.tensor([16, 15, 42, 43, 370, 27, 28, 2]),
            torch.tensor([1, 1, 1, 1, 1, 2, 2, 2]),
        ),
        (
            torch.tensor([14, 15, 16, 370, 17, 18, 19, 20, 370, 370, 2]),
            torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4]),
        ),
    ],
)
def test_target_temporal_ids(
    teach_edh_dataset: TeachEdhDataset,
    target_tokens: torch.Tensor,
    expected_target_tokens: torch.Tensor,
) -> None:
    """Ensure that temporal ids for taget tokens are constructed correctly.

    Separator token id=370
    """
    target_temporal_ids = teach_edh_dataset._make_target_temporal_ids(target_tokens=target_tokens)

    assert torch.equal(target_temporal_ids, expected_target_tokens)
