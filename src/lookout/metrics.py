"""Latency instrumentation, from the first inference call onward.

- prometheus_client Histogram `lookout_inference_seconds`, labels {model, tier}.
- A context manager `timed(model, tier)` wrapping every inference call — tier-1
  frames included, so the tiering claim in the write-up has real numbers.
- start_http_server() for /metrics.

By-host labels, Grafana dashboards, and multi-worker aggregation are documented
future work; the label set is already shaped so adding {host} later is additive.
"""
