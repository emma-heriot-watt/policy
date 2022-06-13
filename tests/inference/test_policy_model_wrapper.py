import pytest
from emma_datasets.datamodels.datasets import TeachEdhInstance
from PIL import Image

from emma_policy.inference.model_wrapper import PolicyModelWrapper


@pytest.mark.skip(reason="Need the checkpoint config.")
def test_model_is_setup_correctly(policy_model_wrapper: PolicyModelWrapper) -> None:
    """Verify the model has been loaded from the checkpoint correctly."""
    # TODO(amit): What other checks should be added?
    assert not policy_model_wrapper._model.train


@pytest.mark.skip(reason="Not implemented yet.")
def test_wrapper_correctly_converts_instance_to_dataset_instance(
    policy_model_wrapper: PolicyModelWrapper,
    teach_edh_instance: TeachEdhInstance,
    teach_edh_instance_history_images: list[Image.Image],
    edh_instance_next_image: Image.Image,
) -> None:
    # Prepare policy model wrapper
    policy_model_wrapper.start_new_edh_instance(
        edh_instance=teach_edh_instance,
        edh_history_images=teach_edh_instance_history_images,
        edh_name=teach_edh_instance.instance_id,
    )

    # next_dataset_instance = policy_model_wrapper._convert_edh_to_dataset_instance(
    #     edh_instance_next_image
    # )

    # TODO(amit): Verify it worked properly.
