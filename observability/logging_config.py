from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone

from observability.request_context import request_id_var, trace_id_var


class JsonFormatter(logging.Formatter):
    """Render one JSON object per log record for CloudWatch ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "service": os.getenv("SERVICE_NAME", "retriva-api"),
            "environment": os.getenv("ENVIRONMENT", "local"),
            "commit_sha": os.getenv("GIT_COMMIT_SHA", ""),
            "trace_id": trace_id_var.get(),
            "request_id": request_id_var.get(),
            "error_class": getattr(record, "error_class", ""),
            "file": getattr(record, "file_path", ""),
            "line": getattr(record, "line_number", 0),
        }

        if record.exc_info:
            exc_type, _exc_value, exc_tb = record.exc_info
            payload["error_class"] = exc_type.__name__
            payload["stack_trace"] = self.formatException(record.exc_info)
            extracted = traceback.extract_tb(exc_tb)
            if extracted:
                payload["file"] = extracted[-1].filename
                payload["line"] = extracted[-1].lineno

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure the root logger to emit JSON lines."""
    root = logging.getLogger()
    root.setLevel((level or os.getenv("LOG_LEVEL", "INFO")).upper())

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
