import logging
from io import BytesIO
from pathlib import Path

from emma_datasets.datamodels.datasets import TeachEdhInstance
from fastapi import HTTPException, status
from PIL import Image

from emma_policy.inference.api.teach_state import TeachDatasetSplit


logger = logging.getLogger("uvicorn.error")


def parse_edh_instance(raw_edh_instance: str) -> TeachEdhInstance:
    """Parse raw EDH instance into structure form."""
    try:
        return TeachEdhInstance.parse_raw(raw_edh_instance)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not parse EDH instance",
        )


def get_edh_history_images_from_dir(
    edh_instance: TeachEdhInstance, data_dir: Path, dataset_split: TeachDatasetSplit
) -> list[Image.Image]:
    """Load the EDH history images from the drive."""
    image_dir = data_dir.joinpath("images", dataset_split, edh_instance.game_id)

    edh_history_images = [
        Image.open(image_dir.joinpath(image_file_name))
        for image_file_name in edh_instance.driver_image_history
    ]

    return edh_history_images


def get_edh_history_images(
    edh_instance: TeachEdhInstance,
    raw_images: list[bytes],
    data_dir: Path,
    dataset_split: TeachDatasetSplit,
) -> list[Image.Image]:
    """Convert the EDH history images from the request to a list of PIL Images.

    The API _should_ be returning a list of images as bytes. These need to be converted back into
    PIL Images so we can do something with them.
    """
    if not edh_instance.driver_image_history:
        return []

    logging.info(f"Attempting to load {len(raw_images)} images from bytes")
    edh_history_images = [Image.open(BytesIO(raw_image)) for raw_image in raw_images]

    if not edh_history_images:
        logger.info("Attempting to load EDH history images from disk")
        edh_history_images = get_edh_history_images_from_dir(edh_instance, data_dir, dataset_split)

    if not edh_history_images:
        logger.error(f"History images are empty for EDH instance `{edh_instance.game_id}`")

    return edh_history_images
