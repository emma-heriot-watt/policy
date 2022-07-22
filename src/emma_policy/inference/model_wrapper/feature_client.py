import io
import logging
from dataclasses import dataclass

import requests
import torch
from PIL import Image
from pydantic import AnyHttpUrl


logger = logging.getLogger(__name__)


@dataclass
class FeatureResponse:
    """Base model for returning bbox features."""

    bbox_features: torch.Tensor
    bbox_coords: torch.Tensor
    bbox_probas: torch.Tensor
    cnn_features: torch.Tensor
    # TODO(amit): Uncomment when stored features contain this property
    # class_labels: list[str]
    width: int
    height: int


class FeatureClient:
    """Simple client for making requests to a feature extraction server."""

    def __init__(self, feature_extractor_endpoint: AnyHttpUrl) -> None:
        self._endpoint = feature_extractor_endpoint

        self._single_feature_endpoint = f"{self._endpoint}/features"
        self._update_model_device_endpoint = f"{self._endpoint}/update_device"

    def update_device(self, device: torch.device) -> None:
        """Update the device used by the feature extractor."""
        logger.info(f"Asking Feature Extractor to move to device: `{device}`")

        response = requests.post(self._update_model_device_endpoint, json={"device": str(device)})

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            logger.info(f"Feature extractor model not moved to device `{device}`")

        if response.status_code == requests.codes.ok:
            logger.info(f"Feature extractor model moved to device `{device}`")

    def post_request(self, image: Image.Image) -> FeatureResponse:
        """Posts a request to the feature extraction server and receive results."""
        image_bytes = self._convert_single_image_to_bytes(image=image)
        request_files = {"input_file": image_bytes}
        response = requests.post(self._single_feature_endpoint, files=request_files, timeout=5)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)

        data = response.json()
        feature_response = FeatureResponse(
            bbox_features=torch.tensor(data["bbox_features"]),
            bbox_coords=torch.tensor(data["bbox_coords"]),
            bbox_probas=torch.tensor(data["bbox_probas"]),
            cnn_features=torch.tensor(data["cnn_features"]),
            # TODO(amit): Uncomment when stored features contain this property
            # class_labels=data["class_labels"],
            width=image.size[0],
            height=image.size[1],
        )

        return feature_response

    def _convert_single_image_to_bytes(self, image: Image.Image) -> bytes:
        """Converts a single PIL Image to bytes."""
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image.format)
        return image_bytes.getvalue()
