from .core.compare import AutoEmulate
from .core.logging_config import _setup_library_logging, configure_logging

# Set up default logging once at package import.
# Adds a NullHandler (Python library best practice) and, for users who have
# not configured logging themselves, a default StreamHandler so that INFO
# messages are visible out of the box.
_setup_library_logging()

__all__ = ["AutoEmulate", "configure_logging"]
