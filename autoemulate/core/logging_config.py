import logging

PACKAGE_LOGGER_NAME = "autoemulate"


def _setup_library_logger() -> logging.Logger:
    """Ensure the package root logger has a NullHandler for library-safe logging."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if len(logger.handlers) == 0:
        logger.addHandler(logging.NullHandler())
    return logger


def get_logger(name: str = PACKAGE_LOGGER_NAME) -> logging.Logger:
    """
    Get a library logger.

    Following Python logging best practices for libraries, this function ensures
    that the package root logger has a NullHandler attached. The application
    developer who uses this library can configure handlers as needed.

    Parameters
    ----------
    name: str
        The name of the logger. Defaults to "autoemulate".

    Returns
    -------
    logging.Logger
        A logger instance for the requested name.
    """
    _setup_library_logger()
    return logging.getLogger(name)


_setup_library_logger()
