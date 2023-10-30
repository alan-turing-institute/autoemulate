import logging
import warnings


def configure_logging(log_to_file=False):
    # Create a logger
    logger = logging.getLogger("autoemulate")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(name)s - %(message)s")
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    # Optionally add a file handler
    if log_to_file:
        fh = logging.FileHandler("autoemulate.log")
        fh.setLevel(logging.INFO)
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
