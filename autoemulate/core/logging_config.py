import logging


def get_logger(name: str = "autoemulate") -> logging.Logger:
    """
    Get a logger with a NullHandler for library use.

    Following Python logging best practices for libraries, this function returns
    a logger that only has a NullHandler attached. The application developer who
    uses this library can configure handlers as needed.

    Parameters
    ----------
    name: str
        The name of the logger. Defaults to "autoemulate".

    Returns
    -------
    logging.Logger
        A logger instance with a NullHandler.
    """
    logger = logging.getLogger(name)
    # Only add NullHandler if no handlers exist
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger
