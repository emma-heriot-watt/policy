import logging
from io import BytesIO
from typing import Any, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from PIL import Image
from uvicorn import Config, Server

from emma_policy.inference.api import (
    ApiSettings,
    ApiStore,
    get_edh_history_images,
    parse_api_args,
    parse_edh_instance,
)
from emma_policy.inference.api.logger import setup_logger
from emma_policy.inference.model_wrapper import PolicyModelWrapper, SimulatorAction


logger = logging.getLogger(__name__)


settings = ApiSettings()
api_store: ApiStore = {}
app = FastAPI()
logger.info("Initializing TEACh API")


@app.on_event("startup")
async def startup_event() -> None:
    """Run specific functions when starting up the API."""
    api_args, model_args = parse_api_args()

    api_store["data_dir"] = api_args.data_dir
    api_store["images_dir"] = api_args.images_dir
    api_store["split"] = api_args.split

    logger.info("Loading model")
    api_store["model"] = PolicyModelWrapper.from_argparse(
        process_index=1, num_processes=1, model_args=model_args
    )
    logging.info("Policy TEACh API is setup!")


@app.get("/")
@app.get("/ping")
@app.get("/test")
async def root(response: Response) -> dict[str, Any]:
    """Ping the API to make sure it is responding."""
    response.status_code = status.HTTP_200_OK
    return {"action": "Look Up", "obj_relative_coord": [0.1, 0.2]}


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(response: Response) -> str:
    """Verify all the APIs are running and working."""
    logger.info("Checking Policy API")
    policy_response = status.HTTP_200_OK
    logger.info(f"Policy API Response: {policy_response}")

    async with httpx.AsyncClient() as client:
        logger.info("Checking Perception API")
        perception_response = (await client.get(settings.feature_extractor_endpoint)).status_code
        logger.info(f"Perception API Response: {perception_response}")

    # Verify all the APIs are available.
    all_passed = all(
        [response == status.HTTP_200_OK for response in (policy_response, perception_response)]
    )

    if not all_passed:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return "failed"

    return "success"


@app.post("/start_new_edh_instance", status_code=status.HTTP_200_OK)
async def start_new_edh_instance(
    edh_name: Optional[str] = Form(...),  # noqa: WPS404
    edh_instance: str = Form(...),  # noqa: WPS404
    edh_history_images: list[UploadFile] = File(...),  # noqa: WPS404
) -> str:
    """Reset the model wrapper to start a new EDH instance."""
    logger.info(f"Starting new EDH instance with name `{edh_name}`")

    parsed_edh_instance = parse_edh_instance(edh_instance)

    logger.debug("Loading PIL images from bytes")
    edh_history_image_bytes = [await raw_file.read() for raw_file in edh_history_images]

    try:
        logger.debug("Attempting to parse images for EDH history")
        parsed_edh_history_images = get_edh_history_images(
            parsed_edh_instance, edh_history_image_bytes, api_store["data_dir"], api_store["split"]
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get EDH history images",
        )

    if not parsed_edh_history_images:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No EDH history images present",
        )

    try:
        logger.debug("Starting a new EDH instance on the model")
        api_store["model"].start_new_edh_instance(parsed_edh_instance, parsed_edh_history_images)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start new EDH instance `{edh_name}`",
        )

    logger.info("Successfully started a new EDH instance on the model")
    return "success"


@app.post("/get_next_action", status_code=status.HTTP_200_OK, response_model=SimulatorAction)
async def get_next_action(
    img_name: Optional[str] = Form(...),  # noqa: WPS404
    edh_name: Optional[str] = Form(...),  # noqa: WPS404
    prev_action: Optional[str] = Form(None),  # noqa: WPS404
    edh_instance: str = Form(...),  # noqa: WPS404
    img: UploadFile = File(...),  # noqa: WPS404
) -> SimulatorAction:
    """Get the next action from the model for the given instance."""
    if not img_name or not edh_instance:
        logger.warning("Either img or edh_instance is None")
        return SimulatorAction(action=None, obj_relative_coord=None)

    parsed_edh_instance = parse_edh_instance(edh_instance)

    logger.info(f"Getting next action for EDH `{parsed_edh_instance.instance_id}`")

    logger.debug("Creating PIL image from the bytes")
    raw_image = await img.read()
    image = Image.open(BytesIO(raw_image))

    logger.debug(f"Previous action: {prev_action}")
    previous_simulator_action = (
        SimulatorAction.parse_raw(prev_action) if prev_action is not None else None
    )

    try:
        logger.debug("Attemtping to get next action from the model")
        action, obj_relative_coord = api_store["model"].get_next_action(
            image, parsed_edh_instance, previous_simulator_action, img_name, edh_name
        )
    except Exception:
        error_message = f"Failed to get next action for EDH with name `{edh_name}"
        logger.error(error_message, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message
        )

    logger.info(f"Returning next action `{action}` (EDH `{parsed_edh_instance.instance_id})`")
    return SimulatorAction(action=action, obj_relative_coord=obj_relative_coord)


def main() -> None:
    """Run the API, exactly the same as the way TEACh does it."""
    server = Server(
        Config(
            "emma_policy.commands.run_teach_api:app",
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
        )
    )

    # Separately adjust the log level for EMMA-related modules
    setup_logger(emma_log_level=settings.log_level)

    server.run()


if __name__ == "__main__":
    main()
