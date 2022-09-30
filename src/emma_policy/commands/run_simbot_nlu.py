import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, TypedDict

from fastapi import FastAPI, Request, Response, status
from pydantic import BaseSettings, FilePath
from transformers import PreTrainedTokenizer
from uvicorn import Config, Server

from emma_policy.datamodules.simbot_nlu_datamodule import prepare_nlu_tokenizer
from emma_policy.inference.api.logger import setup_logger
from emma_policy.inference.api.simbot_state import GenerateRequest
from emma_policy.inference.model_wrapper.simbot_nlu_input_builder import SimBotNLUInputBuilder
from emma_policy.models.simbot_nlu_policy import SimBotNLUEmmaPolicy


logger = logging.getLogger(__name__)


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 6000
    host: str = "0.0.0.0"  # noqa: S104
    log_level: str = "debug"
    model_checkpoint_path: FilePath = Path("storage/model/checkpoints/simbot/nlu.ckpt")
    model_name: str = "heriot-watt/emma-base"
    device: str = "cpu"


class ApiStore(TypedDict, total=False):
    """Common state for the API."""

    input_builder: SimBotNLUInputBuilder
    tokenizer: PreTrainedTokenizer
    model: SimBotNLUEmmaPolicy
    num_beams: int
    no_repeat_ngram_size: int
    max_generated_text_length: int


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

    logger.debug("Preparing the model input")
    # If the environment history is greater than 1,
    # the agent has already clarified or acted.
    if len(simbot_request.environment_history) == 1:
        batch = api_store["input_builder"](simbot_request)
        try:
            action = api_store["model"].inference_step(batch)[0]

        except Exception as err:
            # TODO: report session ID for better debugging
            error_message = f"Failed to get next action for request `{simbot_request}"
            logger.error(error_message, exc_info=err)
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            raise err
    else:
        action = "<act>"
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

    # Separately adjust the log level for EMMA-related modules
    setup_logger(emma_log_level=settings.log_level)

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
