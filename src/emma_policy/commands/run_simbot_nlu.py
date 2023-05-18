import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Literal, TypedDict, Union

import torch
from emma_common.api.instrumentation import instrument_app
from emma_common.aws.cloudwatch import add_cloudwatch_handler_to_logger
from emma_common.datamodels import TorchDataMixin
from emma_common.logging import (
    InstrumentedInterceptHandler,
    logger,
    setup_logging,
    setup_rich_logging,
)
from fastapi import FastAPI, Request, Response, status
from opentelemetry import trace
from pydantic import BaseSettings, FilePath
from transformers import PreTrainedTokenizer
from uvicorn import Config, Server

from emma_policy._version import __version__  # noqa: WPS436
from emma_policy.datamodules.simbot_nlu_datamodule import prepare_nlu_tokenizer
from emma_policy.datamodules.simbot_nlu_dataset import SimBotNLUIntents
from emma_policy.inference.model_wrapper.simbot_nlu_input_builder import SimBotNLUInputBuilder
from emma_policy.inference.model_wrapper.simbot_nlu_output_processor import (
    SimBotNLUPredictionProcessor,
)
from emma_policy.models.simbot_combined_policy import SimBotEmmaCombinedPolicy
from emma_policy.models.simbot_nlu_policy import SimBotNLUEmmaPolicy, postprocess_nlu_output


tracer = trace.get_tracer(__name__)
DEFAULT_ACTION = SimBotNLUIntents.act_one_match.value

NLUModelType = Union[SimBotEmmaCombinedPolicy, SimBotNLUEmmaPolicy]


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 6000
    host: str = "0.0.0.0"  # noqa: S104
    workers: int = 1
    log_level: str = "debug"
    model_checkpoint_path: FilePath = Path("storage/model/checkpoints/simbot/nlu.ckpt")
    model_name: str = "heriot-watt/emma-base"
    model_type: Literal["combined", "standalone"] = "combined"
    device: str = "cpu"
    disable_missing_inventory: bool = False
    enable_prediction_patching: bool = True

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
    model: NLUModelType
    output_processor: SimBotNLUPredictionProcessor
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
) -> NLUModelType:
    """Load an NLU checkpoint."""
    if model_type == "combined":
        model = SimBotEmmaCombinedPolicy(
            model_name=model_name,
            num_beams=api_store["num_beams"],
            max_generated_text_length=api_store["max_generated_text_length"],
        ).load_from_checkpoint(checkpoint_path)
    else:
        model = SimBotNLUEmmaPolicy(
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
    api_store["tokenizer"] = prepare_nlu_tokenizer()
    api_store["input_builder"] = SimBotNLUInputBuilder(
        tokenizer=api_store["tokenizer"],
        device=settings.device,
    )
    api_store["valid_action_types"] = [
        intent.value for intent in SimBotNLUIntents if intent.is_nlu_output
    ]
    api_store["output_processor"] = SimBotNLUPredictionProcessor(
        valid_action_types=api_store["valid_action_types"],
        default_prediction=DEFAULT_ACTION,
        disable_missing_inventory=settings.disable_missing_inventory,
        enable_prediction_patching=settings.enable_prediction_patching,
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

    with tracer.start_as_current_span("Model inference"):
        logger.debug("Preparing the model input")
        # If the environment history is greater than 1,
        # the agent has already clarified or acted.
        if len(simbot_request.environment_history) == 1:
            batch, instruction = api_store["input_builder"](simbot_request)
            try:  # noqa: WPS229
                with torch.no_grad():
                    actions = api_store["model"].inference_step(batch)

                decoded_action = postprocess_nlu_output(api_store["tokenizer"], actions)[0]

                action = api_store["output_processor"](
                    instruction=instruction,
                    prediction=decoded_action,
                    frame_features=simbot_request.environment_history[-1].features,
                )

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
    if settings.traces_to_opensearch:
        instrument_app(
            app,
            otlp_endpoint=settings.otlp_endpoint,
            service_name=settings.opensearch_service_name,
            service_version=__version__,
            service_namespace="SimBot",
        )
        setup_logging(sys.stdout, InstrumentedInterceptHandler())
    else:
        setup_rich_logging(rich_traceback_show_locals=False)

    server = Server(
        Config(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
        )
    )
    if settings.log_to_cloudwatch:
        add_cloudwatch_handler_to_logger(
            boto3_profile_name=settings.aws_profile,
            log_stream_name=settings.watchtower_log_stream_name,
            log_group_name=settings.watchtower_log_group_name,
            send_interval=1,
            enable_trace_logging=settings.traces_to_opensearch,
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
