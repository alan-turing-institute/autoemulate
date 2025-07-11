import logging
import os
import sys
from pathlib import Path


def configure_logging(log_to_file=False, level: str = "INFO"):
    """Configures the logging system.

    Parameters
    ----------
    log_to_file : bool or string, optional
        If True, logs will be written to a file.
        If a string, logs will be written to the specified file.
    verbose : str, optional
        The verbosity level. Can be "error", "warning",
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
        case _:
            msg = 'verbose must be "error", "warning", "info", or "debug"'
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
