from .core.compare import AutoEmulate
from .core.logging_config import _setup_library_logging, configure_logging

# Set up package logging once at import.
# Adds a NullHandler (Python library best practice) and, when the root logger
# is otherwise unconfigured, a lightweight stdout handler for low-code use.
_setup_library_logging()

__all__ = ["AutoEmulate", "configure_logging"]
