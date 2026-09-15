"""The webhook sink against a fake transport, and the metrics endpoint for real
on a spare port."""

import json
import socket

import httpx
from prometheus_client import REGISTRY

from lookout.actions import ActionContext, HAWebhookSink
from lookout.config import ActionSpec
from lookout.metrics import serve, timed

CTX = ActionContext(chain_id="arrivals", camera="driveway", trigger_label="car", trigger_ts=15.53,
                    step_id="confirm-delivery", answer="DELIVERED")


class Hook:
    def __init__(self, status: int = 200):
        self.status = status
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(self.status, text="" if self.status < 300 else "nope")
        return httpx.MockTransport(handle)


def test_webhook_posts_flat_json_with_action_fields_and_context():
    hook = Hook()
    sink = HAWebhookSink("https://ha.test/api/webhook/lookout", transport=hook.transport())
    sink.fire(ActionSpec(type="notify", message="Package delivered"), CTX)
    assert len(sink.fired) == 1 and sink.failed == 0
    req = hook.requests[0]
    assert req.method == "POST" and str(req.url) == "https://ha.test/api/webhook/lookout"
    body = json.loads(req.content)
    assert body["type"] == "notify" and body["message"] == "Package delivered"
    assert body["chain_id"] == "arrivals" and body["answer"] == "DELIVERED" and body["trigger_ts"] == 15.53


def test_webhook_failure_is_counted_not_raised():
    sink = HAWebhookSink("https://ha.test/api/webhook/x", transport=Hook(status=500).transport())
    sink.fire(ActionSpec(type="ha_webhook", service="tv_off"), CTX)
    assert sink.failed == 1 and len(sink.fired) == 1

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    sink = HAWebhookSink("https://ha.test/api/webhook/x", transport=httpx.MockTransport(boom))
    sink.fire(ActionSpec(type="notify", message="m"), CTX)
    assert sink.failed == 1


def test_action_field_named_like_context_is_kept_and_context_prefixed():
    hook = Hook()
    sink = HAWebhookSink("https://ha.test/api/webhook/x", transport=hook.transport())
    sink.fire(ActionSpec(type="notify", camera="front-cam-override"), CTX)
    body = json.loads(hook.requests[0].content)
    assert body["camera"] == "front-cam-override" and body["context_camera"] == "driveway"


def test_metrics_endpoint_serves_the_inference_histogram():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    serve(port, "127.0.0.1")
    with timed("test-model", "tier9"):
        pass
    text = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5).text
    assert 'lookout_inference_seconds_count{model="test-model",tier="tier9"}' in text
    assert REGISTRY.get_sample_value("lookout_inference_seconds_count", {"model": "test-model", "tier": "tier9"}) == 1.0
