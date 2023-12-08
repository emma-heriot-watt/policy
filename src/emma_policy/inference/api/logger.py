import logging
import os
from typing import Union

from loguru import logger
from rich.logging import RichHandler


JSON_LOGS = os.environ.get("JSON_LOGS", "0") == "1"


LOGGER_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


class InterceptHandler(logging.Handler):
    """Logger Handler to intercept log messages from all callers."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit method for logging."""
        # Get corresponding Loguru level if it exists
        level: Union[str, int]

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logger(
    general_log_level: Union[str, int] = "INFO", emma_log_level: Union[str, int] = "INFO"
) -> None:
    """Setup a better logger for the API.

    Log level for emma-related modules can be set differently to the default.
    """
    if isinstance(general_log_level, str):
        general_log_level = general_log_level.upper()

    if isinstance(emma_log_level, str):
        emma_log_level = emma_log_level.upper()

    # intercept everything at the root logger
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(general_log_level)

    # remove every other logger's handlers
    # and propagate to root logger
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

        if name.startswith("emma_"):
            logging.getLogger(name).setLevel(emma_log_level)

    # configure loguru
    logger.configure(
        handlers=[
            {"sink": RichHandler(markup=True), "format": LOGGER_FORMAT, "serialize": JSON_LOGS}
        ]
    )
