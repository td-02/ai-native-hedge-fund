from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

orders_total = Counter(
    "orders_total",
    "Total number of submitted portfolio orders",
    labelnames=("broker", "status"),
)

order_latency_seconds = Histogram(
    "order_latency_seconds",
    "Latency of order submission pipeline",
    labelnames=("broker",),
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

