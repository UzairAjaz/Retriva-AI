from __future__ import annotations

from app.usage import average_cost_per_request


def test_average_cost_per_request() -> None:
    assert average_cost_per_request(10.0, 4) == 2.5
