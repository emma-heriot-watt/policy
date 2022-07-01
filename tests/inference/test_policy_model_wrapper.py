import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from emma_datasets.datamodels.datasets import TeachEdhInstance
from PIL import Image
from requests_mock import Mocker

from emma_policy.inference.api.settings import FeatureExtractorSettings
from emma_policy.inference.model_wrapper import PolicyModelWrapper, SimulatorAction


def load_frame_features_like_api_response(features_path: Path) -> list[dict[str, Any]]:
    """Load the features from the file and convert them to a JSON Serializable form."""
    loaded_frames = torch.load(features_path)["frames"]

    response_features = [
        {
            feature_name: features.tolist() if isinstance(features, torch.Tensor) else features
            for feature_name, features in frame["features"].items()
        }
        for frame in loaded_frames
    ]

    return response_features


def test_model_is_loaded_from_checkpoint(policy_model_wrapper: PolicyModelWrapper) -> None:
    """Verify the model has been loaded from the checkpoint correctly."""
    assert not policy_model_wrapper._model.training


def test_new_edh_instance_is_initialized(
    policy_model_wrapper: PolicyModelWrapper,
    teach_edh_instance: TeachEdhInstance,
    teach_edh_instance_history_images: list[Image.Image],
    requests_mock: Mocker,
) -> None:
    """Verify that a new EDH instance is properly initialized within the wrapper."""
    history_features = load_frame_features_like_api_response(teach_edh_instance.features_path)

    extract_features_endpoint = FeatureExtractorSettings().get_single_feature_url()
    requests_mock.register_uri(
        "POST", extract_features_endpoint, [{"json": features} for features in history_features]
    )

    policy_model_wrapper.start_new_edh_instance(
        edh_instance=teach_edh_instance,
        edh_history_images=teach_edh_instance_history_images,
        edh_name=teach_edh_instance.instance_id,
    )

    assert policy_model_wrapper._edh_instance_state.decoding_step == 1
    assert (
        policy_model_wrapper._teach_edh_inference_dataset.previous_frame
        == teach_edh_instance_history_images[-1]
    )
    assert len(policy_model_wrapper._teach_edh_inference_dataset._feature_dicts) == len(
        teach_edh_instance_history_images
    )


def test_next_action_can_be_predicted(
    policy_model_wrapper: PolicyModelWrapper,
    teach_edh_instance: TeachEdhInstance,
    teach_edh_instance_history_images: list[Image.Image],
    edh_instance_next_image: Image.Image,
    requests_mock: Mocker,
) -> None:
    """Verify that the next action can be predicted after starting a new edh instance."""
    history_features = load_frame_features_like_api_response(teach_edh_instance.features_path)
    future_features = load_frame_features_like_api_response(
        teach_edh_instance.future_features_path
    )

    extract_features_endpoint = FeatureExtractorSettings().get_single_feature_url()
    requests_mock.register_uri(
        "POST",
        extract_features_endpoint,
        [{"json": features} for features in itertools.chain(history_features, future_features)],
    )

    policy_model_wrapper.start_new_edh_instance(
        edh_instance=teach_edh_instance,
        edh_history_images=teach_edh_instance_history_images,
        edh_name=teach_edh_instance.instance_id,
    )

    assert policy_model_wrapper._edh_instance_state.decoding_step == 1
    assert (
        policy_model_wrapper._teach_edh_inference_dataset.previous_frame
        == teach_edh_instance_history_images[-1]
    )
    assert len(policy_model_wrapper._teach_edh_inference_dataset._feature_dicts) == len(
        teach_edh_instance_history_images
    )

    previous_action = None
    previous_state = deepcopy(policy_model_wrapper._edh_instance_state)

    next_action, action_coords = policy_model_wrapper.get_next_action(
        edh_instance_next_image, teach_edh_instance, previous_action
    )
    # Verify the decoding step has increases by 1
    assert (
        policy_model_wrapper._edh_instance_state.decoding_step == previous_state.decoding_step + 1
    )


@pytest.mark.skip(reason="Not all future images have been downloaded/added to the fixtures")
def test_successive_next_actions_can_be_predicted(
    policy_model_wrapper: PolicyModelWrapper,
    teach_edh_instance: TeachEdhInstance,
    teach_edh_instance_history_images: list[Image.Image],
    teach_edh_instance_future_images: list[Image.Image],
    requests_mock: Mocker,
) -> None:
    """Verfiy all successive next actions can be predicted."""
    history_features = load_frame_features_like_api_response(teach_edh_instance.features_path)
    future_features = load_frame_features_like_api_response(
        teach_edh_instance.future_features_path
    )

    extract_features_endpoint = FeatureExtractorSettings().get_single_feature_url()
    requests_mock.register_uri(
        "POST",
        extract_features_endpoint,
        [{"json": features} for features in itertools.chain(history_features, future_features)],
    )

    policy_model_wrapper.start_new_edh_instance(
        edh_instance=teach_edh_instance,
        edh_history_images=teach_edh_instance_history_images,
        edh_name=teach_edh_instance.instance_id,
    )

    assert policy_model_wrapper._edh_instance_state.decoding_step == 1
    assert (
        policy_model_wrapper._teach_edh_inference_dataset.previous_frame
        == teach_edh_instance_history_images[-1]
    )
    assert len(policy_model_wrapper._teach_edh_inference_dataset._feature_dicts) == len(
        teach_edh_instance_history_images
    )

    previous_action = None
    previous_state = deepcopy(policy_model_wrapper._edh_instance_state)

    for future_image in teach_edh_instance_future_images:
        next_action, action_coords = policy_model_wrapper.get_next_action(
            future_image, teach_edh_instance, previous_action
        )

        # Verify the decoding step has increases by 1
        assert (
            policy_model_wrapper._edh_instance_state.decoding_step
            == previous_state.decoding_step + 1
        )

        # Update the state tracking
        previous_action = SimulatorAction(action=next_action, obj_relative_coord=action_coords)
        previous_state = deepcopy(policy_model_wrapper._edh_instance_state)
