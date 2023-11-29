import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from emma_datasets.datamodels.datasets import TeachEdhInstance
from PIL import Image
from requests_mock import Mocker

from emma_policy.api.clients import FeatureExtractorClient
from emma_policy.common.settings import Settings
from emma_policy.inference.model_wrapper import PolicyModelWrapper, SimulatorAction


@pytest.fixture(scope="module")
def single_feature_extractor_endpoint() -> str:
    endpoint = Settings().feature_extractor_endpoint
    feature_client = FeatureExtractorClient(endpoint)
    return feature_client._extract_single_feature_endpoint


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
