import logging
from pathlib import Path
from typing import Any, TypedDict

from fastapi import FastAPI, HTTPException, Response, status
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
    model_name: str = "heriot-watt/emma-base-nlu"
    device: str = "cpu"


class ApiStore(TypedDict, total=False):
    """Common state for the API."""

    input_builder: SimBotNLUInputBuilder
    tokenizer: PreTrainedTokenizer
    model: SimBotNLUEmmaPolicy


settings = ApiSettings()
api_store: ApiStore = {}
app = FastAPI()
logger.info("Initializing Inference API")


def load_model(checkpoint_path: Path, device: str) -> SimBotNLUEmmaPolicy:
    """Load an NLU checkpoint."""
    model = SimBotNLUEmmaPolicy.load_from_checkpoint(checkpoint_path, map_location=device)  # type: ignore[arg-type]
    return model


@app.on_event("startup")
async def startup_event() -> None:
    """Run specific functions when starting up the API."""
    api_store["tokenizer"] = prepare_nlu_tokenizer()
    api_store["input_builder"] = SimBotNLUInputBuilder(tokenizer=api_store["tokenizer"])
    api_store["model"] = load_model(
        checkpoint_path=settings.model_checkpoint_path,
        device=settings.device,
    )
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


@app.post("/generate")
async def generate(request: GenerateRequest) -> str:
    """Get the next action from the model for the given instance."""
    logger.debug("Preparing the model input")
    # If the environment history is greater than 1,
    # the agent has already clarified or acted.
    if len(request.environment_history) == 1:
        batch = api_store["input_builder"](request)
        try:
            action = api_store["model"].inference_step(batch)[0]

        except Exception:
            # TODO: report session ID for better debugging
            error_message = "Failed to get next action."
            logger.error(error_message, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message
            )
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


if __name__ == "__main__":
    main()
