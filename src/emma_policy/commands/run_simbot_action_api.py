import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, Union

import torch
from emma_common.datamodels import TorchDataMixin
from emma_common.logging import logger, setup_rich_logging
from fastapi import FastAPI, Request, Response, status
from pydantic import BaseSettings, FilePath
from transformers import PreTrainedTokenizer
from uvicorn import Config, Server

from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.datamodules.simbot_action_datamodule import prepare_action_tokenizer
from emma_policy.datamodules.simbot_combined_datamodule import prepare_combined_tokenizer
from emma_policy.inference.model_wrapper.simbot_action_input_builder import (
    SimBotActionInputBuilder,
)
from emma_policy.inference.model_wrapper.simbot_action_output_processor import (
    SimBotActionPredictionProcessor,
    SimBotFindPredictionProcessor,
    post_process_action,
)
from emma_policy.models.simbot_combined_policy import SimBotEmmaCombinedPolicy
from emma_policy.models.simbot_emma_policy import SimBotEmmaPolicy


PolicyModelType = Union[SimBotEmmaCombinedPolicy, SimBotEmmaPolicy]


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 6000
    host: str = "0.0.0.0"  # noqa: S104
    workers: int = 1
    log_level: str = "debug"
    model_checkpoint_path: FilePath = Path("storage/model/checkpoints/simbot/action.ckpt")
    model_name: str = "heriot-watt/emma-base"
    model_type: Literal["combined", "standalone"] = "combined"

    device: str = "cpu"
    raw_distance_threshold: int = 2


class ApiStore(TypedDict, total=False):
    """Common state for the API."""

    input_builder: SimBotActionInputBuilder
    tokenizer: PreTrainedTokenizer
    model: PolicyModelType
    action_output_processor: SimBotActionPredictionProcessor
    find_output_processor: SimBotFindPredictionProcessor
    max_length_per_action_sequence: int
    num_beams: int
    no_repeat_ngram_size: int


settings = ApiSettings()
api_store: ApiStore = {}
app = FastAPI()
logger.info("Initializing Inference API")


def load_model(
    checkpoint_path: str,
    model_name: str,
    device: str,
    model_type: Literal["combined", "standalone"],
) -> PolicyModelType:
    """Load a SimBotAction checkpoint."""
    model: PolicyModelType
    if model_type == "combined":
        model = SimBotEmmaCombinedPolicy(model_name=model_name)
    else:
        model = SimBotEmmaPolicy(model_name=model_name)
    model = model.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


@app.on_event("startup")
async def startup_event() -> None:
    """Run specific functions when starting up the API."""
    args = parse_api_args()
    api_store["max_length_per_action_sequence"] = args.max_length_per_action_sequence
    api_store["num_beams"] = args.num_beams
    api_store["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    if settings.model_type == "combined":
        api_store["tokenizer"] = prepare_combined_tokenizer(settings.model_name)
    else:
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
        model_type=settings.model_type,
    )
    logging.info(f"Model is on device: {api_store['model'].device}")

    api_store["action_output_processor"] = SimBotActionPredictionProcessor()
    api_store["find_output_processor"] = SimBotFindPredictionProcessor()

    logging.info("Inference service is setup!")


@app.get("/")
@app.get("/ping")
@app.get("/test")
async def root(response: Response) -> dict[str, Any]:
    """Ping the API to make sure it is responding."""
    response.status_code = status.HTTP_200_OK
    return {"action": "Pickup Bowl <vis_token_42>"}


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(response: Response) -> str:
    """Verify all the APIs are running and working."""
    logger.info("Checking Policy API")
    policy_response = status.HTTP_200_OK
    logger.info(f"Policy API Response: {policy_response}")

    return "success"


@app.post("/generate_find", status_code=status.HTTP_200_OK)
async def generate_find(request: Request, response: Response) -> list[str]:
    """Endpoint for find."""
    try:
        request_body = await request.body()
        simbot_request = TorchDataMixin.get_object(request_body)
    except Exception as request_err:
        logging.exception("Unable to parse request", exc_info=request_err)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise request_err

    (instruction, batch, decoder_input_ids, step_index) = api_store["input_builder"](
        simbot_request, task=Task.visual_grounding
    )

    if batch is not None:
        if decoder_input_ids is not None:
            len_decode = decoder_input_ids.shape[1]
        else:
            len_decode = 0
        try:
            with torch.no_grad():
                model_output = api_store["model"].inference_step(
                    batch,
                    decoder_input_ids=decoder_input_ids,
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

    post_processed_actions = api_store["find_output_processor"](actions, simbot_request)

    logger.debug(f"Predicted actions: {post_processed_actions}")
    return post_processed_actions


@app.post("/grab_from_history", status_code=status.HTTP_200_OK)
async def grab_from_history(request: Request, response: Response) -> Optional[int]:
    """Endpoint for find."""
    try:
        request_body = await request.body()
        simbot_request = TorchDataMixin.get_object(request_body)

    except Exception as request_err:
        logging.exception("Unable to parse request", exc_info=request_err)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise request_err

    (_, batch, decoder_input_ids, step_index) = api_store["input_builder"](
        simbot_request, task=Task.visual_grounding
    )

    if batch is not None:
        len_decode = 0
        try:
            with torch.no_grad():
                model_output = api_store["model"].inference_step(
                    batch,
                    decoder_input_ids=None,
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

    # Select all step_indexes with an object
    filtered_step_idx = [
        step_index[idx] for idx, action in enumerate(actions) if "_token" in action and step_index
    ]
    logger.debug(f"Filtered steps: {filtered_step_idx}")

    unique_ordered_steps = sorted(set(filtered_step_idx))
    logger.debug(f"Sorted ordered steps: {unique_ordered_steps}")

    # most recent timestep with object
    most_recent_step = unique_ordered_steps[-1] if unique_ordered_steps else None

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
        request_body = await request.body()
        simbot_request = TorchDataMixin.get_object(request_body)
    except Exception as request_err:
        logging.exception("Unable to parse request", exc_info=request_err)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise request_err

    # (batch, decoder_input_ids, step_index) = api_store["input_builder"](
    # )
    (raw_input, batch, decoder_input_ids, step_index) = api_store["input_builder"](
        simbot_request, task=Task.action_execution
    )
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
                    num_beams=api_store["num_beams"],
                    no_repeat_ngram_size=api_store["no_repeat_ngram_size"],
                    max_length=max_length,
                )
                action = api_store["tokenizer"].batch_decode(
                    model_output[:, len_decode:], skip_special_tokens=False
                )[0]

                action = api_store["action_output_processor"](
                    prediction=action,
                    frame_features=simbot_request.environment_history[-1].features,
                    instruction=raw_input,
                )

        except Exception as err:
            # TODO: report session ID for better debugging
            error_message = f"Failed to get next action for request `{simbot_request}"
            logger.error(error_message, exc_info=err)
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            raise err
    else:
        action = ""
        logger.debug(f"Empty action for request: {simbot_request}")

    action = post_process_action(action)
    logger.debug(f"Predicted action: {action}")
    return action


def main() -> None:
    """Runs the server."""
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
        default=128,  # noqa: WPS432
        help="Tokenizer maximum number of language tokens",
    )
    arg_parser.add_argument(
        "--max_length_per_action_sequence",
        type=int,
        default=80,  # noqa: WPS432
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
