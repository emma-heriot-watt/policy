import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, TypedDict

from fastapi import FastAPI, Request, Response, status
from pydantic import BaseSettings, FilePath
from transformers import PreTrainedTokenizer
from uvicorn import Config, Server

from emma_policy.datamodules.simbot_action_datamodule import prepare_action_tokenizer
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
    log_level: str = "debug"
    model_checkpoint_path: FilePath = Path("storage/model/checkpoints/simbot/action.ckpt")
    model_name: str = "heriot-watt/emma-base"
    device: str = "cpu"


class ApiStore(TypedDict, total=False):
    """Common state for the API."""

    input_builder: SimBotActionInputBuilder
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

    (batch, decoder_input_ids) = api_store["input_builder"](simbot_request)
    if batch is not None:
        max_length = api_store["max_length_per_action_sequence"]
        if decoder_input_ids is not None:
            max_length += decoder_input_ids.shape[1]
            len_decode = decoder_input_ids.shape[1]
        else:
            len_decode = 0
        try:
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
        default=20,  # noqa: WPS432
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
