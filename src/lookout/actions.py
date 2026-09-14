"""Action sinks.

MockSink (default): logs exactly what it would have done, structured — this is
what the demo and the fixtures run against.

HAWebhookSink (optional): POSTs to the configured Home Assistant webhook URL.
Only active when actions.ha_webhook_url is set in config. Keep it to a single
httpx.post with a timeout; HA-side automation does the rest.
"""
