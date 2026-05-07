import logging
import os
import sys
from pathlib import Path


def _setup_library_logging():
    """
    Set up the library's logger with only a NullHandler.

    This follows Python best practices: libraries should not configure handlers.
    Only NullHandler prevents "No handler could be found" warnings. Handler
    configuration is the responsibility of the application using the library.

    See: https://docs.python.org/3/library/logging.html#configuring-logging-for-a-library
    """
    logger = logging.getLogger("autoemulate")
    # Only add NullHandler if no handlers exist yet
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


# Initialize library logger when module is imported
_setup_library_logging()


def configure_logging(log_to_file=False, level: str = "INFO"):
    """
    Configure the logging system for the autoemulate package.

    This is an OPTIONAL utility for application developers who want to control
    how autoemulate logs messages. It is NOT called automatically by autoemulate.

    Following Python best practices (PEP 391), libraries should not configure
    handlers by default. This function is provided for convenience when you want
    to set up logging for the autoemulate library.

    Parameters
    ----------
    log_to_file: bool or string, optional
        If True, logs will be written to a file named "autoemulate.log" in the
        current working directory. If a string, logs will be written to the
        specified file path. Defaults to False (no file logging).
    level: str, optional
        The logging level. Can be "critical", "error", "warning", "info",
        or "debug". Defaults to "info".

    Returns
    -------
    logging.Logger
        The configured logger instance for "autoemulate".

    Notes
    -----
    If you call this function, it will clear any previously configured handlers
    on the logger and set up new console and optional file handlers.

    Examples
    --------
    >>> from autoemulate.core.logging_config import configure_logging
    >>> logger = configure_logging(level="debug")
    >>> logger = configure_logging(log_to_file=True, level="info")
    >>> logger = configure_logging(log_to_file="/path/to/app.log", level="debug")
    """
    logger = logging.getLogger("autoemulate")
    logger.handlers = []  # Clear existing handlers

    verbose_lower = level.lower()
    match verbose_lower:
        case "error":
            console_log_level = logging.ERROR
        case "warning":
            console_log_level = logging.WARNING
        case "info":
            console_log_level = logging.INFO
        case "debug":
            console_log_level = logging.DEBUG
        case "critical":
            console_log_level = logging.CRITICAL
        case _:
            msg = 'verbose must be "critical", "error", "warning", "info", or "debug"'
            raise ValueError(msg)

    logger.setLevel(console_log_level)

    # Create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_log_level)

    # Create formatter and add it to the handler
    # formatter = logging.Formatter("%(name)s - %(message)s")
    formatter = logging.Formatter("%(levelname)-8s%(asctime)s - %(name)s - %(message)s")
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    # Optionally log to a file
    if log_to_file:
        if isinstance(log_to_file, bool):
            log_file_path = Path.cwd() / "autoemulate.log"
        else:
            log_file_path = Path(log_to_file)
        # Create the directory if it doesn't exist
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        log_file_dir = os.path.dirname(log_file_path)
        if log_file_dir and not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)

        try:
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            logger.exception("Failed to create log file at %s", log_file_path)

    # Capture (model) warnings and redirect them to the logging system
    logging.captureWarnings(True)

    warnings_logger = logging.getLogger("py.warnings")
    for handler in logger.handlers:
        warnings_logger.addHandler(handler)
    warnings_logger.setLevel(logger.getEffectiveLevel())

    return logger
