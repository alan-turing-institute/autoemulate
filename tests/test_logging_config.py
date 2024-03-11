import logging
import sys
import tempfile
import warnings
from pathlib import Path

import pytest

from autoemulate.logging_config import _configure_logging


@pytest.fixture
def cleanup_log_file():
    yield  # let the test run
    log_file_path = Path.cwd() / "autoemulate.log"
    if log_file_path.exists():
        log_file_path.unlink()  # Deletes the log file


def test_configure_logging_without_file_logging():
    logger = _configure_logging(log_to_file=False, verbose=0)

    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1

    console_handler = logger.handlers[0]
    assert isinstance(console_handler, logging.StreamHandler)
    assert console_handler.level == logging.ERROR

    formatter = console_handler.formatter
    assert isinstance(formatter, logging.Formatter)

    assert isinstance(logging.getLogger("py.warnings"), logging.Logger)


def test_configure_logging_with_file_logging(cleanup_log_file):
    log_file = "autoemulate.log"

    logger = _configure_logging(log_to_file=True, verbose=2)

    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 2

    console_handler = logger.handlers[0]
    assert isinstance(console_handler, logging.StreamHandler)
    assert console_handler.level == logging.INFO

    file_handler = logger.handlers[1]
    assert isinstance(file_handler, logging.FileHandler)
    assert file_handler.level == logging.DEBUG

    formatter = file_handler.formatter
    assert isinstance(formatter, logging.Formatter)

    assert isinstance(logging.getLogger("py.warnings"), logging.Logger)


def test_configure_logging_with_invalid_verbose():
    with pytest.raises(ValueError):
        _configure_logging(log_to_file=False, verbose=3)


def test_configure_logging_warnings_logger():
    logger = _configure_logging(log_to_file=False, verbose=0)

    warnings_logger = logging.getLogger("py.warnings")
    assert warnings_logger.level == logger.getEffectiveLevel()


def test_configure_logging_with_file_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "logs" / "autoemulate.log"
        logger = _configure_logging(log_to_file=log_path, verbose=0)

        assert log_path.exists()
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 2
