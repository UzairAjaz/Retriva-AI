from __future__ import annotations

import logging
import os

from observability.logging_config import JsonFormatter
from observability.request_context import RequestContextFilter

_CONFIGURED = False


def configure_cloudwatch_logging() -> bool:
    """Attach a Watchtower handler when explicitly enabled.

    The handler is optional so local development and CI do not require AWS.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return True
    if os.getenv("CLOUDWATCH_ENABLED", "false").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False

    log_group = os.getenv("CLOUDWATCH_LOG_GROUP", "").strip()
    if not log_group:
        logging.getLogger(__name__).warning(
            "CLOUDWATCH_ENABLED is true but CLOUDWATCH_LOG_GROUP is empty"
        )
        return False

    try:
        import boto3
        import watchtower

        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or None
        client = boto3.client("logs", region_name=region)
        handler = watchtower.CloudWatchLogHandler(
            log_group_name=log_group,
            log_stream_name=os.getenv("CLOUDWATCH_LOG_STREAM", "retriva-api-local"),
            boto3_client=client,
            create_log_group=False,
            send_interval=float(os.getenv("CLOUDWATCH_SEND_INTERVAL", "1")),
        )
        handler.setFormatter(JsonFormatter())
        handler.addFilter(RequestContextFilter())
        logging.getLogger().addHandler(handler)
        _CONFIGURED = True
        logging.getLogger(__name__).info(
            "CloudWatch logging enabled", extra={"cloudwatch_log_group": log_group}
        )
        return True
    except Exception:
        logging.getLogger(__name__).exception(
            "CloudWatch logging could not be enabled; using local logs only"
        )
        return False
