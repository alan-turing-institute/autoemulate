import logging
import warnings

PACKAGE_LOGGER_NAME = "autoemulate"
_LEGACY_LOG_LEVELS = {
    "progress_bar",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
}


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


def _warn_deprecated_log_level(log_level: str | None) -> str | None:
    """Validate and warn for the deprecated per-object ``log_level`` API."""
    if log_level is None:
        return None

    legacy_level = log_level.lower()
    if legacy_level not in _LEGACY_LOG_LEVELS:
        msg = (
            f"Invalid log level: {log_level}. Must be one of: "
            f"{sorted(_LEGACY_LOG_LEVELS)}"
        )
        raise ValueError(msg)

    warnings.warn(
        "`log_level` is deprecated and no longer configures logging. Configure "
        "logging from the application that uses autoemulate, and use "
        "`show_progress_bar` to control progress bars.",
        DeprecationWarning,
        stacklevel=3,
    )
    return legacy_level


def _resolve_show_progress_bar(
    log_level: str | None = None,
    show_progress_bar: bool = True,
) -> bool:
    """Resolve legacy ``log_level`` to the new progress-bar flag."""
    _warn_deprecated_log_level(log_level)
    if not isinstance(show_progress_bar, bool):
        msg = "show_progress_bar must be a boolean value."
        raise TypeError(msg)
    return show_progress_bar
