import logging
import sys
import warnings


def _configure_logging(log_to_file=False, verbose=0):
    # Create a logger
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

    # Create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_log_level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(name)s - %(message)s")
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    # Optionally add a file handler
    if log_to_file:
        fh = logging.FileHandler("autoemulate.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Set up the warnings logger to suppress warnings from models we use
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.ERROR)  # Only log errors and above, not warnings
    warnings_logger.addHandler(
        ch
    )  # Still use the console handler for error level warnings

    return logger
