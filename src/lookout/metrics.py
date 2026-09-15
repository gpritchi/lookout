"""Latency instrumentation, from the first inference call onward.

One Histogram, `lookout_inference_seconds`, labelled {model, tier}. Every
inference call — tier-1 detector frames included — runs inside `timed()`, so the
write-up's tiering claim ("tier-1 is milliseconds, tier-2 is seconds") comes with
real numbers rather than an assertion.

The label set is deliberately small. By-host labels, Grafana dashboards, and
multi-worker aggregation are future work; adding {host} later is additive.
`serve()` exposes the registry at /metrics for Prometheus to scrape.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Histogram, start_http_server

log = logging.getLogger(__name__)

# Buckets span two regimes on one axis: detector frames (a few ms to ~100 ms on
# CPU) and VLM calls (seconds to a minute on a busy card). Finer than the
# prometheus default at the top end, where the interesting tier-2 spread lives.
_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0,
)

INFERENCE_SECONDS = Histogram(
    "lookout_inference_seconds",
    "Wall-clock seconds per inference call, by model and tier.",
    labelnames=("model", "tier"),
    buckets=_BUCKETS,
)


def serve(port: int, host: str = "0.0.0.0") -> None:
    """Start the /metrics HTTP server on a daemon thread. prometheus_client
    serves its default registry, which is where INFERENCE_SECONDS lives."""
    start_http_server(port, addr=host)
    log.info("metrics: serving http://%s:%d/metrics", host, port)


@contextmanager
def timed(model: str, tier: str) -> Iterator[None]:
    """Observe the wall-clock duration of the wrapped block. Records on error
    too: a call that raised still cost the card its time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        INFERENCE_SECONDS.labels(model=model, tier=tier).observe(time.perf_counter() - start)
