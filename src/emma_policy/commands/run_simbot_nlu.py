import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, TypedDict

import torch
from emma_common.logging import logger
from fastapi import FastAPI, Request, Response, status
from pydantic import BaseSettings, FilePath
from transformers import PreTrainedTokenizer
from uvicorn import Config, Server

from emma_policy.api.instrumentation import get_tracer
from emma_policy.datamodules.simbot_nlu_datamodule import prepare_nlu_tokenizer
from emma_policy.datamodules.simbot_nlu_dataset import SimBotNLUIntents
from emma_policy.inference.api.simbot_state import GenerateRequest
from emma_policy.inference.model_wrapper.simbot_nlu_input_builder import SimBotNLUInputBuilder
from emma_policy.models.simbot_nlu_policy import SimBotNLUEmmaPolicy


tracer = get_tracer(__name__)
DEFAULT_ACTION = SimBotNLUIntents.act_match.value


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 6000
    host: str = "0.0.0.0"  # noqa: S104
    workers: int = 1
    log_level: str = "debug"
    model_checkpoint_path: FilePath = Path("storage/model/checkpoints/simbot/nlu.ckpt")
    model_name: str = "heriot-watt/emma-base"
    device: str = "cpu"

    # Observability
    traces_to_opensearch: bool = False
    log_to_cloudwatch: bool = False
    aws_profile: str = "TeamProfile"
    watchtower_log_group_name: str = "simbot_challenge"
    watchtower_log_stream_name: str = "nlu/{machine_name}/{logger_name}/{process_id}"

    otlp_endpoint: str = "localhost:4317"
    opensearch_service_name: str = "nlu"


class ApiStore(TypedDict, total=False):
    """Common state for the API."""

    input_builder: SimBotNLUInputBuilder
    tokenizer: PreTrainedTokenizer
    model: SimBotNLUEmmaPolicy
    num_beams: int
    no_repeat_ngram_size: int
    max_generated_text_length: int
    valid_action_types: list[str]


settings = ApiSettings()
api_store: ApiStore = {}
app = FastAPI()
logger.info("Initializing Inference API")


def load_model(checkpoint_path: str, model_name: str, device: str) -> SimBotNLUEmmaPolicy:
    """Load an NLU checkpoint."""
    model = SimBotNLUEmmaPolicy(
        model_name=model_name,
        num_beams=api_store["num_beams"],
        max_generated_text_length=api_store["max_generated_text_length"],
    ).load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def rule_based_ambiguity_check(action: str, frame_features: list[dict[str, Any]]) -> str:
    """Change clarification to action if you can't find multiples of the predicted object."""
    split_parts = action.split(" ")
    object_name = " ".join(split_parts[1:]) if len(split_parts) > 1 else None
    class_labels = frame_features[0].get("class_labels", None)
    if object_name is None or class_labels is None:
        return action
    found_objects = [object_class.lower() == object_name for object_class in class_labels]
    # For now, overwrite the NLU only if there are no multiples in front of you
    # So if there's only one object that you are looking at, assume no ambiguity
    if sum(found_objects) == 1:
        action = DEFAULT_ACTION

    return action


def process_nlu_output(action: str, valid_action_types: list[str]) -> str:
    """Process the NLU output to a valid form."""
    # For search intents only return <search>
    if action.startswith(SimBotNLUIntents.search.value):
        return SimBotNLUIntents.search.value
    # Make sure to return a valid format
    action_type = action.split(" ")[0]
    if action_type not in valid_action_types:
        action = DEFAULT_ACTION
    return action


@app.on_event("startup")
async def startup_event() -> None:
    """Run specific functions when starting up the API."""
    args = parse_api_args()
    api_store["max_generated_text_length"] = args.max_generated_text_length
    api_store["num_beams"] = args.num_beams
    api_store["tokenizer"] = prepare_nlu_tokenizer()
    api_store["input_builder"] = SimBotNLUInputBuilder(
        tokenizer=api_store["tokenizer"],
        device=settings.device,
    )
    api_store["valid_action_types"] = [
        intent.value for intent in SimBotNLUIntents if intent.is_nlu_output
    ]
    logging.info(f"Loading model on device `{settings.device}`")
    api_store["model"] = load_model(
        checkpoint_path=str(settings.model_checkpoint_path),
        model_name=settings.model_name,
        device=settings.device,
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
        simbot_request = GenerateRequest.parse_obj(await request.json())
    except Exception as request_err:
        logging.exception("Unable to parse request", exc_info=request_err)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise request_err

    with tracer.start_as_current_span("Model inference"):
        logger.debug("Preparing the model input")
        # If the environment history is greater than 1,
        # the agent has already clarified or acted.
        if len(simbot_request.environment_history) == 1:
            batch = api_store["input_builder"](simbot_request)
            try:
                with torch.no_grad():
                    action = api_store["model"].inference_step(batch)[0]
                action = process_nlu_output(action, api_store["valid_action_types"])

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
    server = Server(
        Config(
            "emma_policy.commands.run_simbot_nlu:app",
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
