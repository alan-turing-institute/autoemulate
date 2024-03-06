import logging
import sys
import warnings


def _configure_logging(log_to_file=False, verbose=0):
    """Configures the logging system.

    Parameters
    ----------
    log_to_file : bool, optional
        If True, logs will be written to a file. Defaults to False.
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

    # Optionally add a file handler
    if log_to_file:
        fh = logging.FileHandler("autoemulate.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Capture (model) warnings by redirecting them to the logging system
    logging.captureWarnings(True)

    # Redirect warnings to the logging system
    warnings_logger = logging.getLogger("py.warnings")
    for handler in logger.handlers:
        warnings_logger.addHandler(handler)
    warnings_logger.setLevel(logger.getEffectiveLevel())

    return logger
