from __future__ import annotations

from observability.cloudwatch import configure_cloudwatch_logging


def test_cloudwatch_disabled_by_default(monkeypatch):
    monkeypatch.setenv("CLOUDWATCH_ENABLED", "false")
    assert configure_cloudwatch_logging() is False


def test_cloudwatch_enabled_without_log_group_is_safe(monkeypatch):
    monkeypatch.setenv("CLOUDWATCH_ENABLED", "true")
    monkeypatch.setenv("CLOUDWATCH_LOG_GROUP", "")
    assert configure_cloudwatch_logging() is False
