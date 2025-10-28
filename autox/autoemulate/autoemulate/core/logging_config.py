import logging
import os
import sys
from pathlib import Path


def configure_logging(log_to_file=False, level: str = "INFO"):
    """
    Configure the logging system.

    Parameters
    ----------
    log_to_file: bool or string, optional
        If True, logs will be written to a file.
        If a string, logs will be written to the specified file.
    verbose: str, optional
        The verbosity level. Can be "critical", "error", "warning",
          "info", or "debug". Defaults to "info".
    """
    logger = logging.getLogger("autoemulate")
    logger.handlers = []  # Clear existing handlers

    logger.setLevel(logging.DEBUG)

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


def get_configured_logger(
    log_level, progress_bar_attr="progress_bar"
) -> tuple[logging.Logger, bool]:
    """
    Configure logger and progress bar flag consistently.

    Parameters
    ----------
    log_level: str
        The logging level to set. Can be "progress_bar", "debug", "info",
        "warning", "error", or "critical".
    progress_bar_attr: str
        The attribute to check for progress bar. If log_level is set to this value,
        the logger will be set to "error" level and progress_bar will be True. Defaults
        to "progress_bar".

    Returns
    -------
    tuple[logging.Logger, bool]
        The configured logger and the progress bar flag.
    """
    valid_log_levels = [
        "progress_bar",
        "debug",
        "info",
        "warning",
        "error",
        "critical",
    ]
    log_level = log_level.lower()
    if log_level not in valid_log_levels:
        raise ValueError(
            f"Invalid log level: {log_level}. Must be one of: {valid_log_levels}"
        )
    if log_level == progress_bar_attr:
        log_level = "error"
        progress_bar = True
    else:
        progress_bar = False
    logger = configure_logging(level=log_level)
    return logger, progress_bar
