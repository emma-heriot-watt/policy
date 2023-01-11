import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional, TypedDict

import torch
from emma_common.api.instrumentation import instrument_app
from emma_common.aws.cloudwatch import add_cloudwatch_handler_to_logger
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
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.datamodules.simbot_action_datamodule import prepare_action_tokenizer
from emma_policy.inference.api.simbot_state import GenerateRequest
from emma_policy.inference.model_wrapper.simbot_action_input_builder import (
    SimBotActionInputBuilder,
)
from emma_policy.inference.model_wrapper.simbot_raw_text_matcher import SimBotActionRawTextMatcher
from emma_policy.models.simbot_emma_policy import SimBotEmmaPolicy


tracer = trace.get_tracer(__name__)


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 6000
    host: str = "0.0.0.0"  # noqa: S104
    workers: int = 1
    log_level: str = "debug"
    model_checkpoint_path: FilePath = Path("storage/model/checkpoints/simbot/action.ckpt")
    model_name: str = "heriot-watt/emma-base"
    device: str = "cpu"
    raw_text_match_json: Path = Path("storage/constants/simbot_low_level_examples.json")
    raw_distance_threshold: int = 2

    # Observability
    traces_to_opensearch: bool = False
    log_to_cloudwatch: bool = False
    aws_profile: str = "TeamProfile"
    watchtower_log_group_name: str = "simbot_challenge"
    watchtower_log_stream_name: str = "policy/{machine_name}/{logger_name}/{process_id}"

    otlp_endpoint: str = "localhost:4317"
    opensearch_service_name: str = "policy"


class ApiStore(TypedDict, total=False):
    """Common state for the API."""

    input_builder: SimBotActionInputBuilder
    raw_text_matcher: SimBotActionRawTextMatcher
    tokenizer: PreTrainedTokenizer
    model: SimBotEmmaPolicy
    max_length_per_action_sequence: int
    num_beams: int
    no_repeat_ngram_size: int


settings = ApiSettings()
api_store: ApiStore = {}
app = FastAPI()
logger.info("Initializing Inference API")


def load_model(checkpoint_path: str, model_name: str, device: str) -> SimBotEmmaPolicy:
    """Load a SimBotAction checkpoint."""
    model = SimBotEmmaPolicy(model_name=model_name)
    model = model.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def post_process_action(action: str) -> str:
    """Post process the action string.

    Remove the </s><s> at the begining of an instruction. Remove padding tokens. Keep other special
    tokens e.g, <vis_token_5>.
    """
    action = action.lstrip()
    action = action.replace("</s><s>", "")
    action = action.replace("<s>", "")
    action = action.replace("<pad>", "")
    return action


@app.on_event("startup")
async def startup_event() -> None:
    """Run specific functions when starting up the API."""
    args = parse_api_args()
    api_store["max_length_per_action_sequence"] = args.max_length_per_action_sequence
    api_store["num_beams"] = args.num_beams
    api_store["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    api_store["tokenizer"] = prepare_action_tokenizer(settings.model_name)
    api_store["input_builder"] = SimBotActionInputBuilder(
        tokenizer=api_store["tokenizer"],
        device=settings.device,
    )

    logging.info(f"Loading model on device `{settings.device}`")
    api_store["model"] = load_model(
        checkpoint_path=str(settings.model_checkpoint_path),
        model_name=settings.model_name,
        device=settings.device,
    )
    logging.info(f"Model is on device: {api_store['model'].device}")
    logging.info("Inference service is setup!")

    api_store["raw_text_matcher"] = SimBotActionRawTextMatcher(
        raw_text_match_json=settings.raw_text_match_json,
        distance_threshold=settings.raw_distance_threshold,
    )


@app.get("/")
@app.get("/ping")
@app.get("/test")
async def root(response: Response) -> dict[str, Any]:
    """Ping the API to make sure it is responding."""
    response.status_code = status.HTTP_200_OK
    logger.info(response.body)
    return {"action": "Pickup Bowl <vis_token_42>"}


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(response: Response) -> str:
    """Verify all the APIs are running and working."""
    logger.info("Checking Policy API")
    policy_response = status.HTTP_200_OK
    logger.info(f"Policy API Response: {policy_response}")

    return "success"


@app.post("/generate_raw_text_match", status_code=status.HTTP_200_OK)
async def generate_raw_text_match(request: Request, response: Response) -> Optional[str]:
    """Endpoint for simple raw text matching."""
    try:
        simbot_request = GenerateRequest.parse_obj(await request.json())
    except Exception as request_err:
        logging.exception("Unable to parse request", exc_info=request_err)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise request_err
    with tracer.start_as_current_span("Raw text match"):
        output_string = api_store["raw_text_matcher"](simbot_request)
    return output_string


@app.post("/generate_find", status_code=status.HTTP_200_OK)
async def generate_find(request: Request, response: Response) -> list[str]:
    """Endpoint for find."""
    try:
        simbot_request = GenerateRequest.parse_obj(await request.json())
    except Exception as request_err:
        logging.exception("Unable to parse request", exc_info=request_err)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise request_err

    (batch, decoder_input_ids, step_index) = api_store["input_builder"](
        simbot_request, task=Task.visual_grounding
    )
    with tracer.start_as_current_span("Model inference"):
        if batch is not None:
            max_length = api_store["max_length_per_action_sequence"]
            if decoder_input_ids is not None:
                max_length += decoder_input_ids.shape[1]
                len_decode = decoder_input_ids.shape[1]
            else:
                len_decode = 0
            try:
                with torch.no_grad():
                    model_output = api_store["model"].inference_step(
                        batch,
                        decoder_input_ids=decoder_input_ids,
                        max_length=max_length,
                        num_beams=api_store["num_beams"],
                        no_repeat_ngram_size=api_store["no_repeat_ngram_size"],
                    )
                    actions = api_store["tokenizer"].batch_decode(
                        model_output[:, len_decode:], skip_special_tokens=False
                    )

            except Exception as err:
                # TODO: report session ID for better debugging
                error_message = f"Failed to get next action for request `{simbot_request}"
                logger.error(error_message, exc_info=err)
                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                raise err
        else:
            actions = [""]
            logger.debug(f"Empty action for request: {simbot_request}")

    with tracer.start_as_current_span("Post processing"):
        post_processed_actions = []
        for idx, action in enumerate(actions, 1):
            # Append only positive predictions, in case of no object return None
            if "token" in action:
                processed_action = post_process_action(action)
                # Fix the frame token in the case of multiple images
                processed_action = processed_action.replace(
                    "<frame_token_1>", f"<frame_token_{idx}>"
                )
                # Replace the <stop></s> at the end of the prediction
                # We know that the model has finished predicting in visual grounding.
                processed_action = processed_action.replace("<stop></s>", "").strip()
                post_processed_actions.append(processed_action)
    logger.debug(f"Predicted actions: {post_processed_actions}")
    return post_processed_actions


@app.post("/grab_from_history", status_code=status.HTTP_200_OK)
async def grab_from_history(request: Request, response: Response) -> Optional[int]:
    """Endpoint for find."""
    try:
        simbot_request = GenerateRequest.parse_obj(await request.json())
    except Exception as request_err:
        logging.exception("Unable to parse request", exc_info=request_err)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise request_err

    (batch, decoder_input_ids, step_index) = api_store["input_builder"](
        simbot_request, task=Task.visual_grounding
    )
    with tracer.start_as_current_span("Model inference"):
        if batch is not None:
            max_length = api_store["max_length_per_action_sequence"]
            if decoder_input_ids is not None:
                max_length += decoder_input_ids.shape[1]
                len_decode = decoder_input_ids.shape[1]
            else:
                len_decode = 0
            try:
                with torch.no_grad():
                    model_output = api_store["model"].inference_step(
                        batch,
                        decoder_input_ids=None,
                        max_length=max_length,
                        num_beams=api_store["num_beams"],
                        no_repeat_ngram_size=api_store["no_repeat_ngram_size"],
                    )
                    actions = api_store["tokenizer"].batch_decode(
                        model_output[:, len_decode:], skip_special_tokens=False
                    )

            except Exception as err:
                # TODO: report session ID for better debugging
                error_message = f"Failed to get next action for request `{simbot_request}"
                logger.error(error_message, exc_info=err)
                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                raise err
        else:
            actions = [""]
            logger.debug(f"Empty action for request: {simbot_request}")

    with tracer.start_as_current_span("Post processing"):
        # Select all step_indexes with an object
        filtered_step_idx = [
            step_index[idx]
            for idx, action in enumerate(actions)
            if "_token" in action and step_index
        ]
        logger.debug(f"Filtered steps: {filtered_step_idx}")

        unique_ordered_steps = sorted(set(filtered_step_idx))
        logger.debug(f"Sorted ordered steps: {unique_ordered_steps}")

        # most recent timestep with object
        most_recent_step = unique_ordered_steps[0] if unique_ordered_steps else None
    logger.debug(f"most recent step: {most_recent_step}")
    return most_recent_step


@app.post("/generate", status_code=status.HTTP_200_OK)
async def generate(request: Request, response: Response) -> str:
    """Get the next action from the model for the given instruction, question, and answer.

    This is assumed to be called multiple times for a single instruction until the model predicts
    the eos token.
    """
    # Parse the request from the server
    try:
        simbot_request = GenerateRequest.parse_obj(await request.json())
    except Exception as request_err:
        logging.exception("Unable to parse request", exc_info=request_err)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise request_err

    (batch, decoder_input_ids, step_index) = api_store["input_builder"](
        simbot_request, task=Task.action_execution
    )
    with tracer.start_as_current_span("Model inference"):
        if batch is not None:
            max_length = api_store["max_length_per_action_sequence"]
            if decoder_input_ids is not None:
                max_length += decoder_input_ids.shape[1]
                len_decode = decoder_input_ids.shape[1]
            else:
                len_decode = 0
            try:
                with torch.no_grad():
                    model_output = api_store["model"].inference_step(
                        batch,
                        decoder_input_ids=decoder_input_ids,
                        max_length=max_length,
                        num_beams=api_store["num_beams"],
                        no_repeat_ngram_size=api_store["no_repeat_ngram_size"],
                    )
                    action = api_store["tokenizer"].batch_decode(
                        model_output[:, len_decode:], skip_special_tokens=False
                    )[0]

            except Exception as err:
                # TODO: report session ID for better debugging
                error_message = f"Failed to get next action for request `{simbot_request}"
                logger.error(error_message, exc_info=err)
                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                raise err
        else:
            action = ""
            logger.debug(f"Empty action for request: {simbot_request}")

    with tracer.start_as_current_span("Post processing"):
        action = post_process_action(action)
    logger.debug(f"Predicted action: {action}")
    return action


def main() -> None:
    """Runs the server."""
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
        "--tokenizer_truncation_side",
        type=str,
        default="right",
        choices=["left", "right"],
        help="Tokenizer trunction side",
    )
    arg_parser.add_argument(
        "--max_lang_tokens",
        type=int,
        default=64,  # noqa: WPS432
        help="Tokenizer maximum number of language tokens",
    )
    arg_parser.add_argument(
        "--max_length_per_action_sequence",
        type=int,
        default=50,  # noqa: WPS432
        help="Maximum number of generated tokens for each action",
    )
    arg_parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams during decoding",
    )
    arg_parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="Maximum size of repeated ngrams",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
