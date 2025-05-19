"""
💬 chatspec.logger
"""

import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()


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
        return True  # Always return True to indicate the record should be processed


def setup_logging():
    logger = logging.getLogger("chatspec")

    handler = RichHandler(
        level=logging.DEBUG,  # Set the handler level to DEBUG for testing all levels in __main__ block
        console=console,  # Use the console instance
        rich_tracebacks=True,  # Enable rich tracebacks for exceptions
        show_time=False,  # Hide the time column
        show_path=False,  # Hide the path column
        markup=True,  # Essential to interpret the rich markup added by the filter
    )
    formatter = logging.Formatter(
        "| [bold]{name}[/bold] - {message}", style="{"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RichMarkupFilter())
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("chatspec")

    logger.setLevel(logging.DEBUG)
    print("Testing logging levels with RichHandler and custom markup:")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
