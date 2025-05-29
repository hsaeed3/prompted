"""prompted.logger"""

import logging
from rich.logging import RichHandler
from rich.console import Console
from typing import Literal

# NOTE:
# as you can see this library is heavily opinionated
# that does not mean it is not fast.
# we are speed.

_console = Console()


class RichMarkupFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.CRITICAL:
            record.msg = f"[bold red]{record.msg}[/bold red]"
        elif record.levelno >= logging.ERROR:
            record.msg = f"[italic red]{record.msg}[/italic red]"
        elif record.levelno >= logging.WARNING:
            record.msg = f"[italic yellow]{record.msg}[/italic yellow]"
        elif record.levelno >= logging.INFO:
            record.msg = f"[white]{record.msg}[/white]"
        elif record.levelno >= logging.DEBUG:
            record.msg = f"[italic dim]{record.msg}[/italic dim]"
        return True  


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("prompted")

    handler = RichHandler(
        level=logging.WARNING,  
        console=_console,  
        rich_tracebacks=True,
        show_time=False,
        show_path=False,
        markup=True,
    )
    formatter = logging.Formatter("| [bold]{name}[/bold] - {message}", style="{")
    handler.setFormatter(formatter)
    handler.addFilter(RichMarkupFilter())
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def _get_logger(module : str | None = None) -> logging.Logger:
    if module is None:
        return _logger
    return _logger.getChild(module) # type: ignore


def verbosity(
    level: Literal["debug", "info", "warning", "error", "critical"],
) -> None:
    logger = _get_logger()
    logger.setLevel(level.upper())
    # Update all handlers' levels to match
    for handler in logger.handlers:
        handler.setLevel(level.upper())


_logger = _setup_logging()
"""Singleton logger for the prompted library."""


__all__ = [
    "verbosity",
    "_get_logger",
]