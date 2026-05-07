"""Tests for autoemulate.core.logging_config."""

import logging
import sys
from unittest.mock import patch

import pytest
from autoemulate.core.logging_config import (
    _has_real_handler,
    _parse_level,
    _setup_library_logging,
    configure_logging,
    get_configured_logger,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_autoemulate_logger():
    """Restore the "autoemulate" logger to a clean state before and after each
    test so that handler state does not leak between tests.
    """
    logger = logging.getLogger("autoemulate")
    warnings_logger = logging.getLogger("py.warnings")

    orig_handlers = logger.handlers[:]
    orig_level = logger.level
    orig_propagate = logger.propagate
    orig_warnings_handlers = warnings_logger.handlers[:]
    orig_warnings_level = warnings_logger.level
    orig_warnings_propagate = warnings_logger.propagate

    # Give each test a blank slate
    logger.handlers = []
    warnings_logger.handlers = []
    logging.captureWarnings(False)

    yield logger

    # Close any file handlers created during the test
    for h in list(logger.handlers) + list(warnings_logger.handlers):
        if hasattr(h, "close"):
            h.close()

    logger.handlers = orig_handlers
    logger.level = orig_level
    logger.propagate = orig_propagate
    warnings_logger.handlers = orig_warnings_handlers
    warnings_logger.level = orig_warnings_level
    warnings_logger.propagate = orig_warnings_propagate
    logging.captureWarnings(False)


# ---------------------------------------------------------------------------
# _parse_level
# ---------------------------------------------------------------------------


class TestParseLevel:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("critical", logging.CRITICAL),
            ("CRITICAL", logging.CRITICAL),
            ("error", logging.ERROR),
            ("ERROR", logging.ERROR),
            ("warning", logging.WARNING),
            ("WARNING", logging.WARNING),
            ("info", logging.INFO),
            ("INFO", logging.INFO),
            ("debug", logging.DEBUG),
            ("DEBUG", logging.DEBUG),
        ],
    )
    def test_valid_levels(self, name, expected):
        assert _parse_level(name) == expected

    def test_invalid_level_raises_value_error(self):
        with pytest.raises(ValueError, match="level must be one of"):
            _parse_level("verbose")

    def test_invalid_level_error_contains_input(self):
        with pytest.raises(ValueError, match="badlevel"):
            _parse_level("badlevel")


# ---------------------------------------------------------------------------
# _has_real_handler
# ---------------------------------------------------------------------------


class TestHasRealHandler:
    def test_false_with_only_null_handler(self, isolated_autoemulate_logger):
        isolated_autoemulate_logger.handlers = [logging.NullHandler()]
        assert _has_real_handler(isolated_autoemulate_logger) is False

    def test_false_with_no_handlers(self, isolated_autoemulate_logger):
        isolated_autoemulate_logger.handlers = []
        assert _has_real_handler(isolated_autoemulate_logger) is False

    def test_true_with_stream_handler(self, isolated_autoemulate_logger):
        isolated_autoemulate_logger.handlers = [logging.StreamHandler()]
        assert _has_real_handler(isolated_autoemulate_logger) is True

    def test_true_with_null_and_stream_handler(self, isolated_autoemulate_logger):
        isolated_autoemulate_logger.handlers = [
            logging.NullHandler(),
            logging.StreamHandler(),
        ]
        assert _has_real_handler(isolated_autoemulate_logger) is True


# ---------------------------------------------------------------------------
# _setup_library_logging
# ---------------------------------------------------------------------------


class TestSetupLibraryLogging:
    def test_adds_null_handler(self, isolated_autoemulate_logger):
        _setup_library_logging()
        null_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.NullHandler)
        ]
        assert len(null_handlers) == 1

    def test_no_duplicate_null_handler_on_repeat_calls(
        self, isolated_autoemulate_logger
    ):
        _setup_library_logging()
        _setup_library_logging()
        null_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.NullHandler)
        ]
        assert len(null_handlers) == 1

    def test_adds_stream_handler_when_root_unconfigured(
        self, isolated_autoemulate_logger
    ):
        # Patch root.handlers to simulate an unconfigured logging environment.
        # pytest's own logging plugin adds a handler to the root logger, which
        # would otherwise prevent _setup_library_logging from adding the
        # default StreamHandler.
        with patch.object(logging.root, "handlers", []):
            _setup_library_logging()
        stream_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.NullHandler)
        ]
        assert len(stream_handlers) == 1

    def test_stream_handler_targets_stdout(self, isolated_autoemulate_logger):
        with patch.object(logging.root, "handlers", []):
            _setup_library_logging()
        stream_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.NullHandler)
        ]
        assert stream_handlers[0].stream is sys.stdout

    def test_sets_propagate_false_when_stream_handler_added(
        self, isolated_autoemulate_logger
    ):
        with patch.object(logging.root, "handlers", []):
            _setup_library_logging()
        stream_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.NullHandler)
        ]
        if stream_handlers:
            assert isolated_autoemulate_logger.propagate is False

    def test_sets_level_to_info_when_stream_handler_added(
        self, isolated_autoemulate_logger
    ):
        with patch.object(logging.root, "handlers", []):
            _setup_library_logging()
        assert isolated_autoemulate_logger.level == logging.INFO

    def test_no_stream_handler_when_root_has_handlers(
        self, isolated_autoemulate_logger
    ):
        """When the root logger already has a handler, no stream handler is added
        (the application's own configuration should be respected).
        """
        root = logging.root
        orig_root_handlers = root.handlers[:]
        root.handlers = [logging.StreamHandler()]
        try:
            _setup_library_logging()
            stream_handlers = [
                h
                for h in isolated_autoemulate_logger.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.NullHandler)
            ]
            assert len(stream_handlers) == 0
        finally:
            root.handlers = orig_root_handlers

    def test_no_duplicate_stream_handler_when_logger_already_configured(
        self, isolated_autoemulate_logger
    ):
        """If the autoemulate logger already has a real handler, _setup_library_logging
        must not add another one.
        """
        existing = logging.StreamHandler()
        isolated_autoemulate_logger.handlers = [existing]
        _setup_library_logging()
        stream_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.NullHandler)
        ]
        # Only the pre-existing handler; no duplicate
        assert stream_handlers == [existing]

    def test_no_duplicate_stream_handler_on_repeat_calls(
        self, isolated_autoemulate_logger
    ):
        """A second call to _setup_library_logging() must not add a second StreamHandler
        because the logger already has a real handler after the first call.
        """
        with patch.object(logging.root, "handlers", []):
            _setup_library_logging()  # First call: adds StreamHandler
            _setup_library_logging()  # Second call: sees existing real handler, skips
        stream_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.NullHandler)
        ]
        assert len(stream_handlers) == 1


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_returns_autoemulate_logger(self, isolated_autoemulate_logger):
        result = configure_logging()
        assert isinstance(result, logging.Logger)
        assert result.name == "autoemulate"

    def test_default_adds_stream_handler(self, isolated_autoemulate_logger):
        configure_logging()
        stream_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.NullHandler)
        ]
        assert len(stream_handlers) == 1

    def test_stream_handler_targets_stdout(self, isolated_autoemulate_logger):
        configure_logging()
        stream_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.NullHandler)
        ]
        assert stream_handlers[0].stream is sys.stdout

    @pytest.mark.parametrize(
        "level_str,expected",
        [
            ("debug", logging.DEBUG),
            ("info", logging.INFO),
            ("warning", logging.WARNING),
            ("error", logging.ERROR),
            ("critical", logging.CRITICAL),
        ],
    )
    def test_level_parameter_sets_logger_level(
        self, isolated_autoemulate_logger, level_str, expected
    ):
        configure_logging(level=level_str)
        assert isolated_autoemulate_logger.level == expected

    def test_invalid_level_raises_value_error(self, isolated_autoemulate_logger):
        with pytest.raises(ValueError):
            configure_logging(level="verbose")

    def test_sets_propagate_false(self, isolated_autoemulate_logger):
        isolated_autoemulate_logger.propagate = True
        configure_logging()
        assert isolated_autoemulate_logger.propagate is False

    def test_replaces_handlers_on_repeat_call(self, isolated_autoemulate_logger):
        """Calling configure_logging() twice must not accumulate stream handlers."""
        configure_logging(level="debug")
        configure_logging(level="info")
        stream_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.NullHandler)
        ]
        assert len(stream_handlers) == 1

    def test_disable_removes_non_null_handlers(self, isolated_autoemulate_logger):
        configure_logging()
        configure_logging(disable=True)
        non_null = [
            h
            for h in isolated_autoemulate_logger.handlers
            if not isinstance(h, logging.NullHandler)
        ]
        assert len(non_null) == 0

    def test_disable_re_enables_propagation(self, isolated_autoemulate_logger):
        configure_logging()  # sets propagate=False
        configure_logging(disable=True)
        assert isolated_autoemulate_logger.propagate is True

    def test_disable_resets_level_to_notset(self, isolated_autoemulate_logger):
        configure_logging(level="error")
        configure_logging(disable=True)
        assert isolated_autoemulate_logger.level == logging.NOTSET

    def test_disable_preserves_null_handler(self, isolated_autoemulate_logger):
        isolated_autoemulate_logger.addHandler(logging.NullHandler())
        configure_logging(disable=True)
        null_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.NullHandler)
        ]
        assert len(null_handlers) >= 1

    def test_disable_keeps_user_handlers(self, isolated_autoemulate_logger):
        user_handler = logging.StreamHandler()
        isolated_autoemulate_logger.addHandler(user_handler)
        configure_logging(level="info")
        configure_logging(disable=True)
        assert user_handler in isolated_autoemulate_logger.handlers

    def test_log_to_file_true_adds_file_handler(
        self, isolated_autoemulate_logger, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        configure_logging(log_to_file=True)
        file_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        for h in file_handlers:
            h.close()

    def test_log_to_file_true_creates_autoemulate_log_file(
        self, isolated_autoemulate_logger, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        configure_logging(log_to_file=True)
        assert (tmp_path / "autoemulate.log").exists()
        for h in isolated_autoemulate_logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.close()

    def test_log_to_file_custom_path(self, isolated_autoemulate_logger, tmp_path):
        custom_path = tmp_path / "subdir" / "custom.log"
        configure_logging(log_to_file=str(custom_path))
        assert custom_path.exists()
        for h in isolated_autoemulate_logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.close()

    def test_log_to_file_adds_both_stream_and_file_handler(
        self, isolated_autoemulate_logger, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        configure_logging(log_to_file=True)
        stream_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        file_handlers = [
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1
        assert len(file_handlers) == 1
        for h in file_handlers:
            h.close()

    def test_repeat_call_closes_previous_file_handler(
        self, isolated_autoemulate_logger, tmp_path
    ):
        log_path = tmp_path / "reconfigure.log"
        configure_logging(log_to_file=str(log_path))
        old_file_handler = next(
            h
            for h in isolated_autoemulate_logger.handlers
            if isinstance(h, logging.FileHandler)
        )

        configure_logging(log_to_file=str(log_path))

        assert old_file_handler.stream is None

    def test_info_message_emitted_at_info_level(
        self, isolated_autoemulate_logger, capsys
    ):
        configure_logging(level="info")
        logging.getLogger("autoemulate").info("test-info-message")
        captured = capsys.readouterr()
        assert "test-info-message" in captured.out

    def test_debug_message_suppressed_at_info_level(
        self, isolated_autoemulate_logger, capsys
    ):
        configure_logging(level="info")
        logging.getLogger("autoemulate").debug("test-debug-message")
        captured = capsys.readouterr()
        assert "test-debug-message" not in captured.out

    def test_debug_message_emitted_at_debug_level(
        self, isolated_autoemulate_logger, capsys
    ):
        configure_logging(level="debug")
        logging.getLogger("autoemulate").debug("test-debug-message")
        captured = capsys.readouterr()
        assert "test-debug-message" in captured.out

    def test_does_not_capture_python_warnings(self, isolated_autoemulate_logger):
        warnings_logger = logging.getLogger("py.warnings")
        sentinel_handler = logging.StreamHandler()
        warnings_logger.handlers = [sentinel_handler]

        configure_logging()

        assert warnings_logger.handlers == [sentinel_handler]
        assert getattr(logging, "_warnings_showwarning", None) is None

    def test_formatter_includes_level_and_message(
        self, isolated_autoemulate_logger, capsys
    ):
        configure_logging(level="info")
        logging.getLogger("autoemulate").info("formatted-check")
        captured = capsys.readouterr()
        assert "INFO" in captured.out
        assert "formatted-check" in captured.out


# ---------------------------------------------------------------------------
# get_configured_logger
# ---------------------------------------------------------------------------


class TestGetConfiguredLogger:
    def test_returns_autoemulate_logger(self):
        logger, _ = get_configured_logger("info")
        assert logger.name == "autoemulate"

    @pytest.mark.parametrize(
        "level_str,expected_level",
        [
            ("debug", logging.DEBUG),
            ("info", logging.INFO),
            ("warning", logging.WARNING),
            ("error", logging.ERROR),
            ("critical", logging.CRITICAL),
        ],
    )
    def test_sets_logger_level(self, level_str, expected_level):
        logger, progress_bar = get_configured_logger(level_str)
        assert logger.level == expected_level
        assert progress_bar is False

    def test_progress_bar_returns_flag_true_and_error_level(self):
        logger, progress_bar = get_configured_logger("progress_bar")
        assert progress_bar is True
        assert logger.level == logging.ERROR

    def test_invalid_level_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            get_configured_logger("verbose")

    def test_case_insensitive_level(self):
        logger, progress_bar = get_configured_logger("INFO")
        assert logger.level == logging.INFO
        assert progress_bar is False

    def test_progress_bar_case_insensitive(self):
        _, progress_bar = get_configured_logger("PROGRESS_BAR")
        assert progress_bar is True

    def test_returns_tuple(self):
        result = get_configured_logger("info")
        assert isinstance(result, tuple)
        assert len(result) == 2
