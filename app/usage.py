from __future__ import annotations


def average_cost_per_request(total_cost_usd: float, request_count: int) -> float:
    """Return the average model cost for one request."""
    return total_cost_usd / request_count
