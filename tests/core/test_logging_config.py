import logging

import pytest
from autoemulate.core.logging_config import _resolve_show_progress_bar, get_logger


def test_get_logger_adds_only_null_handler_to_package_logger():
    package_logger = logging.getLogger("autoemulate")
    original_handlers = list(package_logger.handlers)
    try:
        package_logger.handlers = []

        logger = get_logger("autoemulate.tests")

        assert logger.name == "autoemulate.tests"
        assert len(package_logger.handlers) == 1
        assert isinstance(package_logger.handlers[0], logging.NullHandler)
    finally:
        package_logger.handlers = original_handlers


@pytest.mark.parametrize(
    ("log_level", "expected"),
    [
        ("progress_bar", True),
        ("error", True),
    ],
)
def test_resolve_show_progress_bar_warns_for_legacy_log_level(log_level, expected):
    with pytest.warns(DeprecationWarning, match="`log_level` is deprecated"):
        assert _resolve_show_progress_bar(log_level=log_level) is expected


def test_explicit_show_progress_bar_overrides_legacy_log_level():
    with pytest.warns(DeprecationWarning, match="`log_level` is deprecated"):
        show_progress_bar = _resolve_show_progress_bar(
            log_level="progress_bar", show_progress_bar=False
        )
    assert show_progress_bar is False


def test_resolve_show_progress_bar_rejects_none():
    with pytest.raises(TypeError, match="show_progress_bar must be a boolean"):
        _resolve_show_progress_bar(show_progress_bar=None)


def test_resolve_show_progress_bar_rejects_invalid_legacy_log_level():
    with pytest.raises(ValueError, match="Invalid log level"):
        _resolve_show_progress_bar(log_level="verbose")
