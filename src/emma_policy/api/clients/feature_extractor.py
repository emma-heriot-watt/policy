import logging
from io import BytesIO
from typing import Literal, Union

import numpy as np
import requests
import torch
from numpy.typing import ArrayLike
from PIL import Image
from pydantic import AnyHttpUrl, BaseModel


logger = logging.getLogger(__name__)


class FeatureResponse(BaseModel, arbitrary_types_allowed=True):
    """Extracted features from an image."""

    bbox_features: torch.Tensor
    bbox_coords: torch.Tensor
    bbox_probas: torch.Tensor
    cnn_features: torch.Tensor


class FeatureExtractorClient:
    """API Client for making requests to the feature extractor server."""

    def __init__(self, feature_extractor_server_endpoint: AnyHttpUrl) -> None:
        self._endpoint = feature_extractor_server_endpoint

        self._healthcheck_endpoint = f"{self._endpoint}/ping"
        self._extract_single_feature_endpoint = f"{self._endpoint}/features"
        self._extract_batch_features_endpoint = f"{self._endpoint}/batch_features"
        self._update_model_device_endpoint = f"{self._endpoint}/update_device"

    def healthcheck(self) -> bool:
        """Verify the feature extractor server is healthy."""
        response = requests.get(self._healthcheck_endpoint)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            return False

        return True

    def update_device(self, device: torch.device) -> None:
        """Change the device used by the feature extractor.

        This is primarily useful for ensuring the perception and policy model are on the same GPU.
        """
        logger.info(f"Asking Feature Extractor to move to device: `{device}`")

        response = requests.post(self._update_model_device_endpoint, json={"device": str(device)})

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            logger.info(f"Feature extractor model not moved to device `{device}`")

        if response.status_code == requests.codes.ok:
            logger.info(f"Feature extractor model moved to device `{device}`")

    def extract_single_image(self, image: Union[Image.Image, ArrayLike]) -> FeatureResponse:
        """Submit a request to the feature extraction server for a single image."""
        image_bytes = self._convert_single_image_to_bytes(image)
        request_files = {"imput_file": image_bytes}
        response = requests.post(
            self._extract_single_feature_endpoint, files=request_files, timeout=5
        )

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
        )

        return feature_response

    def extract_batch_images(
        self, images: Union[list[Image.Image], list[ArrayLike]]
    ) -> list[FeatureResponse]:
        """Send a batch of images to be extracted by the server.

        There is no batch size limit for the client to send, as the server will extract the maximum
        number of images as it can at any one time.
        """
        all_images_bytes: list[bytes] = [
            self._convert_single_image_to_bytes(image) for image in images
        ]
        request_files: list[tuple[Literal["images"], tuple[str, bytes]]] = [
            ("images", (f"file{idx}", image_bytes))
            for idx, image_bytes in enumerate(all_images_bytes)
        ]

        response = requests.post(
            self._extract_batch_features_endpoint, files=request_files, timeout=5
        )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)

        data = response.json()

        all_feature_responses = [
            FeatureResponse(
                bbox_features=torch.tensor(response_data["bbox_features"]),
                bbox_coords=torch.tensor(response_data["bbox_coords"]),
                bbox_probas=torch.tensor(response_data["bbox_probas"]),
                cnn_features=torch.tensor(response_data["cnn_features"]),
            )
            for response_data in data
        ]

        return all_feature_responses

    def _convert_single_image_to_bytes(self, image: Union[Image.Image, ArrayLike]) -> bytes:
        """Converts a single image to bytes."""
        image_bytes = BytesIO()

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        image.save(image_bytes, format=image.format)
        return image_bytes.getvalue()
