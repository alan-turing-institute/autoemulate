from .core.compare import AutoEmulate
from .core.logging_config import _setup_library_logging, configure_logging

# Set up package logging once at import.
# Adds only a NullHandler (Python library best practice). Applications that
# want visible output can opt in via configure_logging() or their own logging.
_setup_library_logging()

__all__ = ["AutoEmulate", "configure_logging"]
