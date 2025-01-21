import logging
import os
import sys
from pathlib import Path


def _configure_logging(log_to_file=False, verbose=0):
    """Configures the logging system.

    Parameters
    ----------
    log_to_file : bool or string, optional
        If True, logs will be written to a file.
        If a string, logs will be written to the specified file.
    verbose : int, optional
        The verbosity level. Can be 0, 1, or 2. Defaults to 0.
    """

    logger = logging.getLogger("autoemulate")
    logger.handlers = []  # Clear existing handlers

    logger.setLevel(logging.DEBUG)

    match verbose:
        case 0:
            console_log_level = logging.ERROR
        case 1:
            console_log_level = logging.WARNING
        case 2:
            console_log_level = logging.INFO
        case _:
            raise ValueError("verbose must be 0, 1, or 2")

    # Create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_log_level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(name)s - %(message)s")
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
            logger.exception(f"Failed to create log file at {log_file_path}")

    # Capture (model) warnings and redirect them to the logging system
    logging.captureWarnings(True)

    warnings_logger = logging.getLogger("py.warnings")
    for handler in logger.handlers:
        warnings_logger.addHandler(handler)
    warnings_logger.setLevel(logger.getEffectiveLevel())

    return logger
