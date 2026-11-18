from __future__ import annotations

import io
import json
import logging

from observability.logging_config import JsonFormatter
from observability.request_context import (
    RequestContextFilter,
    request_id_var,
    trace_id_var,
)


def _logger_with_stream() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("retriva.tests.logging")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger, stream


def test_json_formatter_includes_required_fields(monkeypatch):
    monkeypatch.setenv("SERVICE_NAME", "retriva-api")
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("GIT_COMMIT_SHA", "abc123")
    request_id_var.set("req-1")
    trace_id_var.set("trace-1")
    try:
        logger, stream = _logger_with_stream()
        logger.info("hello")
        payload = json.loads(stream.getvalue().strip())
    finally:
        request_id_var.set("")
        trace_id_var.set("")

    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["service"] == "retriva-api"
    assert payload["environment"] == "staging"
    assert payload["commit_sha"] == "abc123"
    assert payload["request_id"] == "req-1"
    assert payload["trace_id"] == "trace-1"
    assert payload["error_class"] == ""
    assert payload["file"] == ""
    assert payload["line"] == 0


def test_context_filter_captures_ids_for_async_handlers(monkeypatch):
    monkeypatch.setenv("SERVICE_NAME", "retriva-api")
    request_id_var.set("req-async")
    trace_id_var.set("trace-async")
    record = logging.LogRecord(
        name="retriva.tests",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="async handler record",
        args=(),
        exc_info=None,
    )
    RequestContextFilter().filter(record)
    request_id_var.set("")
    trace_id_var.set("")

    payload = json.loads(JsonFormatter().format(record))
    assert payload["request_id"] == "req-async"
    assert payload["trace_id"] == "trace-async"


def test_json_formatter_captures_exception_location(monkeypatch):
    monkeypatch.setenv("SERVICE_NAME", "retriva-api")
    logger, stream = _logger_with_stream()

    try:
        raise KeyError("boom")
    except KeyError:
        logger.exception("failed")

    payload = json.loads(stream.getvalue().strip())
    assert payload["level"] == "ERROR"
    assert payload["error_class"] == "KeyError"
    assert payload["file"].endswith("test_logging_config.py")
    assert payload["line"] > 0
    assert "KeyError" in payload["stack_trace"]
