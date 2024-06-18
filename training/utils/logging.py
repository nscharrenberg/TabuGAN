from enum import Enum
from typer import colors, style, echo


class LogLevel(Enum):
    ERROR = colors.RED
    INFO = colors.CYAN
    WARNING = colors.YELLOW
    SUCCESS = colors.GREEN
    DEFAULT = colors.WHITE


def log(message: str, level: LogLevel = LogLevel.DEFAULT, verbose: bool = True):
    """
    When allowed, print a log with the given message and log level.

    :param message: The message to log
    :param level: The level of the log (DEBUG = 0, OK = 10, INFO = 20, WARN = 30, ERROR = 40, CRITICAL = 50)
    :param verbose: Whether to print to console or not.
    """
    if not verbose:
        return

    echo(style(message, fg=level.value))
