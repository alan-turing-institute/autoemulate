import logging
import sys
from pathlib import Path

_LOGGER_NAME = "autoemulate"
_MANAGED_HANDLER_ATTR = "_autoemulate_managed"
_FORMATTER = logging.Formatter("%(levelname)-8s%(asctime)s - %(name)s - %(message)s")
_VALID_LEVELS = ("critical", "error", "warning", "info", "debug")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_level(level: str) -> int:
    """Convert a level name string to a :mod:`logging` integer constant."""
    level_lower = level.lower()
    match level_lower:
        case "critical":
            return logging.CRITICAL
        case "error":
            return logging.ERROR
        case "warning":
            return logging.WARNING
        case "info":
            return logging.INFO
        case "debug":
            return logging.DEBUG
        case _:
            msg = f'level must be one of {_VALID_LEVELS}, got "{level}"'
            raise ValueError(msg)


def _has_real_handler(logger: logging.Logger) -> bool:
    """Return True if *logger* has any handler other than NullHandler."""
    return any(not isinstance(h, logging.NullHandler) for h in logger.handlers)


def _ensure_null_handler(logger: logging.Logger) -> None:
    """Ensure the package logger has exactly one NullHandler fallback."""
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())


def _is_managed_handler(handler: logging.Handler) -> bool:
    """Return True if *handler* was created by autoemulate."""
    return getattr(handler, _MANAGED_HANDLER_ATTR, False)


def _mark_managed(handler: logging.Handler) -> logging.Handler:
    """Mark *handler* as owned by autoemulate and return it."""
    setattr(handler, _MANAGED_HANDLER_ATTR, True)
    return handler


def _has_managed_handler(logger: logging.Logger) -> bool:
    """Return True if *logger* has any autoemulate-managed handler."""
    return any(_is_managed_handler(handler) for handler in logger.handlers)


def _remove_managed_handlers(logger: logging.Logger) -> None:
    """Remove and close handlers previously created by autoemulate."""
    for handler in list(logger.handlers):
        if _is_managed_handler(handler):
            logger.removeHandler(handler)
            handler.close()


def _create_stream_handler() -> logging.StreamHandler:
    """Create the default stdout handler used by autoemulate."""
    handler = logging.StreamHandler(sys.stdout)
    _mark_managed(handler)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_FORMATTER)
    return handler


def _create_file_handler(log_file_path: Path) -> logging.FileHandler:
    """Create a managed file handler for *log_file_path*."""
    handler = logging.FileHandler(log_file_path)
    _mark_managed(handler)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_FORMATTER)
    return handler


def _setup_library_logging() -> None:
    """Set up the "autoemulate" logger for use as a library.

    Called **once** when the package is imported.  Follows the Python
    recommendation of adding a NullHandler so that log records are silently
    discarded when no application-level handler is present.

    For users who have *not* configured any logging themselves (e.g. a plain
    script or a Jupyter notebook with a fresh kernel), a default
    StreamHandler is also added so that INFO-level messages are visible
    without any extra setup.  If the root logger already has handlers at
    import time the default StreamHandler is skipped, allowing the
    application's own logging configuration to take full effect via normal
    log propagation.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    _ensure_null_handler(logger)

    # Only install our default StreamHandler if:
    #   1. the root logger has no handlers (logging is unconfigured), and
    #   2. the autoemulate logger does not already have a real handler.
    # This avoids interfering with application-level logging in tests or
    # well-configured programs while still giving novice users useful output.
    if not logging.root.handlers and not _has_real_handler(logger):
        logger.addHandler(_create_stream_handler())
        logger.setLevel(logging.INFO)
        # Prevent messages from being passed on to the root logger as well,
        # which would cause duplicate output if the root logger later gets a
        # StreamHandler (e.g. after the application calls basicConfig).
        logger.propagate = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    log_to_file: bool | str = False,
    level: str = "INFO",
    disable: bool = False,
) -> logging.Logger:
    """Configure logging for the autoemulate package.

    Call this function if you want to customise how the package-wide
    ``"autoemulate"`` logger emits log messages. Advanced users who manage
    their own logging configuration can call
    ``configure_logging(disable=True)`` to remove the handlers installed by
    autoemulate and restore normal root-level propagation.

    Parameters
    ----------
    log_to_file : bool or str, optional
        If True, write logs to ``autoemulate.log`` in the current working
        directory.  If a string, write to the specified file path.
        Defaults to False.
    level : str, optional
        Minimum log level to emit.  Must be one of "critical", "error",
        "warning", "info", or "debug".  Defaults to "INFO".
    disable : bool, optional
        If True, remove only the handlers installed by autoemulate, reset the
        package logger level to inherit from the root logger, and re-enable
        propagation so that the calling application's own logging
        configuration takes effect. Defaults to False.

    Returns
    -------
    logging.Logger
        The "autoemulate" logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    _ensure_null_handler(logger)
    _remove_managed_handlers(logger)

    if disable:
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
        return logger

    level_value = _parse_level(level)
    logger.setLevel(level_value)
    logger.propagate = False

    logger.addHandler(_create_stream_handler())

    if log_to_file:
        log_file_path = (
            Path.cwd() / "autoemulate.log"
            if isinstance(log_to_file, bool)
            else Path(log_to_file)
        )
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.addHandler(_create_file_handler(log_file_path))
        except Exception:
            logger.exception("Failed to create log file at %s", log_file_path)

    return logger


def get_configured_logger(
    log_level: str,
    progress_bar_attr: str = "progress_bar",
) -> tuple[logging.Logger, bool]:
    """Return the package-wide logger and a progress-bar flag.

    Sets the effective level of the shared ``"autoemulate"`` logger according
    to *log_level* but does **not** add or remove any handlers. Handler setup
    is performed once at package import by _setup_library_logging() and can be
    customised at any time via the public configure_logging() API.

    Parameters
    ----------
    log_level : str
        One of "progress_bar", "debug", "info", "warning", "error", or
        "critical".  When "progress_bar" is given the logger level is set
        to ERROR (to suppress verbose output) and the returned flag is True.
    progress_bar_attr : str
        The sentinel value that activates the progress bar.  Defaults to
        "progress_bar".

    Returns
    -------
    tuple[logging.Logger, bool]
        The "autoemulate" logger and whether a progress bar should be shown.
    """
    valid_log_levels = [progress_bar_attr, *_VALID_LEVELS]
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

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_parse_level(log_level))
    return logger, progress_bar
