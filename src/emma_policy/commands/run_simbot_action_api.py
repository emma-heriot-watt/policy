import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, TypedDict

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseSettings
from transformers import AutoTokenizer, PreTrainedTokenizer
from uvicorn import Config, Server

from emma_policy.inference.api.logger import setup_logger
from emma_policy.inference.api.simbot_state import GenerateRequest
from emma_policy.inference.model_wrapper.simbot_action_input_builder import (
    SimBotActionInputBuilder,
)
from emma_policy.models.simbot_emma_policy import SimBotEmmaPolicy


logger = logging.getLogger(__name__)


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 6000
    host: str = "0.0.0.0"  # noqa: S104
    log_level: str = "info"
    device: str = "cpu"


class ApiStore(TypedDict, total=False):
    """Common state for the API."""

    input_builder: SimBotActionInputBuilder
    tokenizer: PreTrainedTokenizer
    model: SimBotEmmaPolicy
    max_length_per_action_sequence: int


settings = ApiSettings()
api_store: ApiStore = {}
app = FastAPI()
logger.info("Initializing Inference API")


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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.truncation_side = args.tokenizer_truncation_side
    if args.max_lang_tokens:
        tokenizer.model_max_length = args.max_lang_tokens

    api_store["max_length_per_action_sequence"] = args.max_length_per_action_sequence
    api_store["tokenizer"] = tokenizer
    api_store["input_builder"] = SimBotActionInputBuilder(
        tokenizer=tokenizer,
        device=settings.device,
    )
    api_store["model"] = SimBotEmmaPolicy.load_from_checkpoint(
        args.model_path, map_location=settings.device
    )

    logging.info("Inference service is setup!")


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


@app.post("/generate", status_code=status.HTTP_200_OK)
async def generate(
    request: GenerateRequest,
) -> str:
    """Get the next action from the model for the given instruction, question, and answer.

    This is assumed to be called multiple times for a single instruction until the model predicts
    the eos token.
    """
    (batch, decoder_input_ids) = api_store["input_builder"](request)
    if batch is not None:
        max_length = api_store["max_length_per_action_sequence"]
        if decoder_input_ids is not None:
            max_length += decoder_input_ids.shape[0]
            len_decode = decoder_input_ids.shape[1]
        else:
            len_decode = 0
        try:
            model_output = api_store["model"].inference_step(
                batch,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
            )
            action = api_store["tokenizer"].batch_decode(
                model_output[:, len_decode:], skip_special_tokens=False
            )[0]

        except Exception:
            # TODO: report session ID for better debugging
            error_message = f"Failed to get next action for request `{request}"
            logger.error(error_message, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message
            )
    else:
        action = ""
        logger.debug(f"Empty action for request: {request}")
    action = post_process_action(action)
    return action


def main() -> None:
    """Runs the server."""
    server = Server(
        Config(
            "emma_policy.commands.run_simbot_action_api:app",
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
        )
    )

    # Separately adjust the log level for EMMA-related modules
    setup_logger(emma_log_level=settings.log_level)

    server.run()


def parse_api_args() -> Namespace:
    """Parse any arguments."""
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to SimBotAction model checkpoint",
    )

    arg_parser.add_argument(
        "--model_name",
        type=Path,
        required=True,
        help="Path to SimBotAction model checkpoint",
    )

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
        default=20,  # noqa: WPS432
        help="Maximum number of generated tokens for each action",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
