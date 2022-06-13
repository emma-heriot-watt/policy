import json
from io import BytesIO
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

from emma_datasets.datamodels.datasets import TeachEdhInstance
from fastapi.testclient import TestClient
from PIL import Image
from pytest_cases import fixture

from emma_policy.commands.run_teach_api import app
from emma_policy.inference.model_wrapper import PolicyModelWrapper


@fixture(scope="module")
def edh_instance_path(fixtures_root: Path) -> Path:
    """Get and return the path to the EDH instance."""
    return fixtures_root.joinpath(
        "teach_edh", "edh_instances", "train", "1c70e34df85e61c8_6282.edh1.json"
    )


@fixture(scope="module")
def inference_images_path(fixtures_root: Path) -> Path:
    """Get the path to where the images are kept."""
    return fixtures_root.joinpath("teach_edh", "inference_images")


@fixture(scope="module")
def teach_edh_instance(edh_instance_path: Path) -> TeachEdhInstance:
    """Get the TEACh EDH Instance used for the tests."""
    return TeachEdhInstance.parse_file(edh_instance_path)


@fixture(scope="module")
def edh_instance_next_image(
    teach_edh_instance: TeachEdhInstance, inference_images_path: Path
) -> Image.Image:
    """Load the next frame that the agent would be given."""
    next_image_name = teach_edh_instance.driver_images_future[0]
    next_image_path = inference_images_path.joinpath(next_image_name)
    image = Image.open(next_image_path)

    return image


@fixture(scope="module")
def teach_edh_instance_history_images(
    teach_edh_instance: TeachEdhInstance, inference_images_path: Path
) -> list[Image.Image]:
    """Convert the driver history images into a list of PIL images.

    Note: InferenceRunner provides a list of `PIL.Image.Image`
    """
    images = []

    for image_file_name in teach_edh_instance.driver_image_history:
        image_path = inference_images_path.joinpath(image_file_name)
        original_image = Image.open(image_path)
        images.append(original_image.copy())
        original_image.close()

    return images


@fixture
def policy_model_wrapper(fixtures_root: Path) -> PolicyModelWrapper:
    """Create a policy model wrapper so no need to keep repeating the args."""
    model_checkpoint_path = fixtures_root.joinpath("teach_tiny.ckpt")
    model_wrapper = PolicyModelWrapper(
        process_index=1,
        num_processes=1,
        model_checkpoint_path=model_checkpoint_path,
        model_name="heriot-watt/emma-tiny",
    )

    return model_wrapper


@fixture(scope="module")
def client(fixtures_root: Path) -> Generator[TestClient, None, None]:
    """Get an API client which can be used for testing."""
    data_dir = fixtures_root.joinpath("teach_edh")
    images_dir = fixtures_root.joinpath("teach_edh", "inference_images")
    split = "train"

    patched_argv = ["main", "--data_dir", data_dir, "--images_dir", images_dir, "--split", split]

    with patch("sys.argv", patched_argv):
        yield TestClient(app)


@fixture(scope="module")
def start_new_edh_instance_request_body(edh_instance_path: Path) -> dict[str, Any]:
    """Get an example request body the API should be able to receive.

    This has been adapted from the `RemoteModel` class in `alexa/teach`:
    https://github.com/alexa/teach/blob/2e5be94ebdef4910a61cb1bce069d80b0079d1d3/src/teach/inference/remote_model.py#L93-L94
    """
    raw_edh_instance = json.loads(edh_instance_path.read_bytes())

    request_body = {
        "edh_name": raw_edh_instance.get("instance_id", None),
        "edh_instance": json.dumps(raw_edh_instance),
    }

    return request_body


@fixture(scope="module")
def start_new_edh_instance_request_files(
    teach_edh_instance_history_images: list[Image.Image],
) -> list[tuple[str, tuple[str, BytesIO, str]]]:
    """Convert images into expected format.

    This has been taken from `RemoteModel` class in `alexa/teach`.
    """
    images = []
    idx = 0

    for image in teach_edh_instance_history_images:
        image_in_memory = BytesIO()
        image.save(image_in_memory, "jpeg")
        image_in_memory.seek(0)
        images.append(("edh_history_images", (f"history{idx}", image_in_memory, "image/jpeg")))
        idx += 1

    return images
