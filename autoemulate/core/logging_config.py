import logging
import sys
from pathlib import Path

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
    logger = logging.getLogger("autoemulate")
    logger.addHandler(logging.NullHandler())

    # Only install our default StreamHandler if:
    #   1. the root logger has no handlers (logging is unconfigured), and
    #   2. the autoemulate logger does not already have a real handler.
    # This avoids interfering with application-level logging in tests or
    # well-configured programs while still giving novice users useful output.
    if not logging.root.handlers and not _has_real_handler(logger):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)  # Logger level controls filtering
        ch.setFormatter(_FORMATTER)
        logger.addHandler(ch)
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

    Call this function if you want to customise how autoemulate emits log
    messages.  Advanced users who manage their own logging configuration can
    call ``configure_logging(disable=True)`` to remove all
    autoemulate-specific handlers and restore normal log propagation so that
    their own setup takes full effect.

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
        If True, remove all autoemulate-specific handlers and re-enable
        propagation to the root logger so that the calling application's own
        logging configuration takes effect.  Defaults to False.

    Returns
    -------
    logging.Logger
        The "autoemulate" logger.
    """
    logger = logging.getLogger("autoemulate")

    # Remove all non-NullHandler handlers previously added by autoemulate.
    logger.handlers = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]

    if disable:
        logger.propagate = True
        return logger

    level_value = _parse_level(level)
    logger.setLevel(level_value)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)  # Handler passes everything; logger level filters
    ch.setFormatter(_FORMATTER)
    logger.addHandler(ch)

    if log_to_file:
        log_file_path = (
            Path.cwd() / "autoemulate.log"
            if isinstance(log_to_file, bool)
            else Path(log_to_file)
        )
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(_FORMATTER)
            logger.addHandler(fh)
        except Exception:
            logger.exception("Failed to create log file at %s", log_file_path)

    # Redirect Python warnings into the logging system.
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers = [
        h for h in warnings_logger.handlers if isinstance(h, logging.NullHandler)
    ]
    for handler in logger.handlers:
        if not isinstance(handler, logging.NullHandler):
            warnings_logger.addHandler(handler)
    warnings_logger.setLevel(logger.level)

    return logger


def get_configured_logger(
    log_level: str,
    progress_bar_attr: str = "progress_bar",
) -> tuple[logging.Logger, bool]:
    """Return the "autoemulate" logger and a progress-bar flag.

    Sets the logger's effective level according to *log_level* but does
    **not** add or remove any handlers.  Handler setup is performed once at
    package import by _setup_library_logging() and can be customised at any
    time via the public configure_logging() API.

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

    logger = logging.getLogger("autoemulate")
    logger.setLevel(_parse_level(log_level))
    return logger, progress_bar
