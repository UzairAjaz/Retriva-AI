from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from observability.request_context import (
    RequestContextMiddleware,
    request_id_var,
    trace_id_var,
)


def _client() -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/context")
    def context():
        return {
            "request_id": request_id_var.get(),
            "trace_id": trace_id_var.get(),
        }

    return TestClient(app)


def test_middleware_propagates_incoming_ids():
    response = _client().get(
        "/context",
        headers={"X-Request-ID": "req-123", "X-Trace-ID": "trace-456"},
    )
    assert response.status_code == 200
    assert response.json() == {"request_id": "req-123", "trace_id": "trace-456"}
    assert response.headers["X-Request-ID"] == "req-123"
    assert response.headers["X-Trace-ID"] == "trace-456"


def test_middleware_generates_ids_when_absent():
    response = _client().get("/context")
    body = response.json()
    assert body["request_id"]
    assert body["trace_id"] == body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert response.headers["X-Trace-ID"] == body["trace_id"]
