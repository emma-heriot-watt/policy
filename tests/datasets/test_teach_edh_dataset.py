from pathlib import Path
from typing import Any
from unittest.mock import PropertyMock

from emma_datasets.datamodels.datasets.teach import TeachEdhInstance
from pytest_cases import fixture
from pytest_mock import MockerFixture

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem
from emma_policy.datamodules.teach_edh_dataset import TeachEdhDataset
from emma_policy.models.tokenizer_emma import EmmaTokenizer


class TeachEdhInstanceFeaturesPathPropertyMock(PropertyMock):  # type: ignore[misc]
    """Mock the `features_path` property within the TeachEdhInstance.

    The features path within each instance is derived automatically and NOT hard-coded into the
    class. Therefore to be able to test the instances properly, we need to override the property
    and return something else.
    """

    def __get__(self, obj: TeachEdhInstance, obj_type: Any = None) -> Path:  # noqa: WPS110
        """Get the features path from the fixtures.

        This updates the `return_value`, which is used by `unittest.Mock` to return a value.
        """
        dataset_index = obj._features_path.parts.index("datasets")
        self.return_value = Path("storage", "fixtures", *obj._features_path.parts[dataset_index:])
        return self()


@fixture
def teach_edh_dataset(
    teach_edh_instances_db: Path, emma_tokenizer: EmmaTokenizer, mocker: MockerFixture
) -> TeachEdhDataset:
    """Instantiate the TeachEdhDataset for each test.

    Additionally, this fixture also mocks the features path of each TeachEdhInstance to point to
    the fixtures dir.
    """
    mocker.patch.object(
        TeachEdhInstance,
        "features_path",
        new_callable=TeachEdhInstanceFeaturesPathPropertyMock,
    )

    return TeachEdhDataset(dataset_db_path=teach_edh_instances_db, tokenizer=emma_tokenizer)


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


def test_utterances_are_immediately_after_the_text_action(
    teach_edh_dataset: TeachEdhDataset,
) -> None:
    """Test the utterances are properly included within the input text.

    This tests verifies the `_convert_actions_to_tokenizable_strings` private method, to verify
    that when the action history from each is given, the utterances are also included within the
    output.
    """
    with teach_edh_dataset.db:
        instance_str: str = teach_edh_dataset.db[0]
    instance = TeachEdhInstance.parse_raw(instance_str)

    # Verify that the driver has dialogue actions within this instance
    assert instance.driver_dialog_history_cleaned

    input_actions_list = teach_edh_dataset._convert_actions_to_tokenizable_strings(
        instance.extended_driver_action_history
    )

    utterance_counter = 0

    for idx, action_string in enumerate(input_actions_list):

        if action_string == "Text":
            assert input_actions_list[idx + 1].startswith("<<")
            assert input_actions_list[idx + 1].endswith(
                instance.driver_dialog_history_cleaned[utterance_counter]
            )

            utterance_counter += 1
