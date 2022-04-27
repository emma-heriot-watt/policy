from io import BytesIO
from typing import Any

import pytest
from fastapi.testclient import TestClient


def test_ping_works(client: TestClient) -> None:
    """Verify the API can be pinged."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"action": "Look Up", "obj_relative_coord": [0.1, 0.2]}


@pytest.mark.skip()
def test_get_edh_history_images_convert_bytes_to_pillow_images() -> None:
    raise NotImplementedError


@pytest.mark.skip()
def test_start_new_instance_prepares_the_model_properly(
    client: TestClient,
    start_new_edh_instance_request_body: dict[str, Any],
    start_new_edh_instance_request_files: list[tuple[str, tuple[str, BytesIO, str]]],
) -> None:
    response = client.post(
        "/start_new_edh_instance",
        data=start_new_edh_instance_request_body,
        files=start_new_edh_instance_request_files,
    )

    assert response.status_code == 200


@pytest.mark.skip(reason="Not implemented yet")
def test_get_next_action_returns_dict_for_the_inference_runner(client: TestClient) -> None:
    raise NotImplementedError
