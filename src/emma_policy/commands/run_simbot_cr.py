import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Literal, TypedDict, Union

import torch
from emma_common.datamodels import TorchDataMixin
from emma_common.logging import logger, setup_rich_logging
from fastapi import FastAPI, Request, Response, status
from pydantic import BaseSettings, FilePath
from transformers import PreTrainedTokenizer
from uvicorn import Config, Server

from emma_policy.datamodules.simbot_combined_datamodule import prepare_combined_tokenizer
from emma_policy.datamodules.simbot_cr_datamodule import prepare_cr_tokenizer
from emma_policy.datamodules.simbot_cr_dataset import SimBotCRIntents
from emma_policy.inference.model_wrapper.simbot_cr_input_builder import SimBotCRInputBuilder
from emma_policy.inference.model_wrapper.simbot_cr_output_processor import (
    SimBotCRPredictionProcessor,
)
from emma_policy.models.simbot_combined_policy import SimBotEmmaCombinedPolicy
from emma_policy.models.simbot_cr_policy import SimBotCREmmaPolicy, postprocess_cr_output


DEFAULT_ACTION = SimBotCRIntents.act_one_match.value

CRModelType = Union[SimBotEmmaCombinedPolicy, SimBotCREmmaPolicy]


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 6000
    host: str = "0.0.0.0"  # noqa: S104
    workers: int = 1
    log_level: str = "debug"
    model_checkpoint_path: FilePath = Path("storage/model/checkpoints/simbot/cr.ckpt")
    model_name: str = "heriot-watt/emma-base"
    model_type: Literal["combined", "standalone"] = "combined"
    device: str = "cpu"
    disable_missing_inventory: bool = False


class ApiStore(TypedDict, total=False):
    """Common state for the API."""

    input_builder: SimBotCRInputBuilder
    tokenizer: PreTrainedTokenizer
    model: CRModelType
    output_processor: SimBotCRPredictionProcessor
    num_beams: int
    no_repeat_ngram_size: int
    max_generated_text_length: int
    valid_action_types: list[str]


settings = ApiSettings()
api_store: ApiStore = {}
app = FastAPI()
logger.info("Initializing Inference API")


def load_model(
    checkpoint_path: str,
    model_name: str,
    device: str,
    model_type: Literal["combined", "standalone"],
) -> CRModelType:
    """Load an CR checkpoint."""
    if model_type == "combined":
        model = SimBotEmmaCombinedPolicy(
            model_name=model_name,
            num_beams=api_store["num_beams"],
            max_generated_text_length=api_store["max_generated_text_length"],
        ).load_from_checkpoint(checkpoint_path)
    else:
        model = SimBotCREmmaPolicy(
            model_name=model_name,
            num_beams=api_store["num_beams"],
            max_generated_text_length=api_store["max_generated_text_length"],
        ).load_from_checkpoint(checkpoint_path)

    model.to(device)
    model.eval()
    return model


@app.on_event("startup")
async def startup_event() -> None:
    """Run specific functions when starting up the API."""
    args = parse_api_args()
    api_store["max_generated_text_length"] = args.max_generated_text_length
    api_store["num_beams"] = args.num_beams
    if settings.model_type == "combined":
        api_store["tokenizer"] = prepare_combined_tokenizer(settings.model_name)
    else:
        api_store["tokenizer"] = prepare_cr_tokenizer(settings.model_name)
    api_store["input_builder"] = SimBotCRInputBuilder(
        tokenizer=api_store["tokenizer"],
        device=settings.device,
    )
    api_store["valid_action_types"] = [
        intent.value for intent in SimBotCRIntents if intent.is_cr_output
    ]
    api_store["output_processor"] = SimBotCRPredictionProcessor(
        valid_action_types=api_store["valid_action_types"],
        default_prediction=DEFAULT_ACTION,
        disable_missing_inventory=settings.disable_missing_inventory,
    )
    logging.info(f"Loading model on device `{settings.device}`")
    api_store["model"] = load_model(
        checkpoint_path=str(settings.model_checkpoint_path),
        model_name=settings.model_name,
        device=settings.device,
        model_type=settings.model_type,
    )
    logging.info(f"Model is on device: {api_store['model'].device}")
    logging.info("Inference service is setup!")


@app.get("/")
@app.get("/ping")
@app.get("/test")
async def root(response: Response) -> dict[str, Any]:
    """Ping the API to make sure it is responding."""
    response.status_code = status.HTTP_200_OK
    return {"action": "act"}


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(response: Response) -> str:
    """Verify all the APIs are running and working."""
    response.status_code = status.HTTP_200_OK
    return "success"


@app.post("/generate", status_code=status.HTTP_200_OK)
async def generate(request: Request, response: Response) -> str:
    """Get the next action from the model for the given instance."""
    # Parse the request from the server
    try:
        request_body = await request.body()
        simbot_request = TorchDataMixin.get_object(request_body)
    except Exception as request_err:
        logging.exception("Unable to parse request", exc_info=request_err)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise request_err

    logger.debug("Preparing the model input")
    # If the environment history is greater than 1,
    # the agent has already clarified or acted.
    if len(simbot_request.environment_history) == 1:
        batch, instruction = api_store["input_builder"](simbot_request)
        try:  # noqa: WPS229
            with torch.no_grad():
                actions = api_store["model"].inference_step(batch)

            decoded_action = postprocess_cr_output(api_store["tokenizer"], actions)[0]

            action = api_store["output_processor"](prediction=decoded_action)

        except Exception as err:
            # TODO: report session ID for better debugging
            error_message = f"Failed to get next action for request `{simbot_request}"
            logger.error(error_message, exc_info=err)
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            raise err
    else:
        action = DEFAULT_ACTION
    return action


def main() -> None:
    """Runs a server that serves any instance of an EMMA policy model."""
    setup_rich_logging(rich_traceback_show_locals=False)

    server = Server(
        Config(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
        )
    )
    server.run()


def parse_api_args() -> Namespace:
    """Parse any arguments."""
    arg_parser = ArgumentParser()
    # TODO: move this to an inference config
    arg_parser.add_argument(
        "--max_generated_text_length",
        type=int,
        default=8,
        help="Maximum number of generated tokens for each action",
    )
    arg_parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams during decoding",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
