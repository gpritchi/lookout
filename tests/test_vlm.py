"""The OpenAI-compatible client: request shape per payload kind, routing by
model name, error handling. All against an in-process fake transport."""

import base64
import json

import httpx
import pytest

from lookout.config import ModelSpec
from lookout.engine import StepPayload
from lookout.events import Frame
from lookout.scheduler import InferenceJob
from lookout.vlm import OpenAICompatibleClient, VlmClientError

JPEG = b"\xff\xd8\xff\xe0fakejpeg"
MODELS = {
    "vlm": ModelSpec(endpoint="http://proxy.test/v1", model="qwen3-vl", capabilities=["image", "image_sequence"]),
    "omni": ModelSpec(endpoint="http://other.test/v1/", model="nemotron", capabilities=["video_clip"]),
}


def frames(n: int, start: float = 10.0, step: float = 2.0) -> tuple[Frame, ...]:
    return tuple(Frame(camera="cam", ts=start + i * step, data=JPEG, ref=f"f{i}") for i in range(n))


def payload(kind: str, n: int, **kw) -> StepPayload:
    return StepPayload(kind=kind, prompt="Which?", frames=frames(n), camera="cam", chain_id="c", step_id="s",
                       generation=1, center_ts=10.0, **kw)


def job(model: str, p: StepPayload) -> InferenceJob:
    return InferenceJob(model=model, priority=1, chain_id="c", step_id="s", payload=p)


class Recorder:
    """httpx transport that records the request and answers with a canned body."""

    def __init__(self, status: int = 200, content: str = "GREY_VAN", raw: bytes | None = None):
        self.status, self.content, self.raw = status, content, raw
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if self.raw is not None:
                return httpx.Response(self.status, content=self.raw)
            body = {"choices": [{"message": {"content": self.content}}], "usage": {"prompt_tokens": 966}}
            return httpx.Response(self.status, json=body)
        return httpx.MockTransport(handle)


def test_image_request_has_one_image_then_prompt():
    rec = Recorder()
    client = OpenAICompatibleClient(MODELS, transport=rec.transport())
    assert client(job("vlm", payload("image", 1))) == "GREY_VAN"
    req = rec.requests[0]
    assert str(req.url) == "http://proxy.test/v1/chat/completions"
    body = json.loads(req.content)
    assert body["model"] == "qwen3-vl" and body["temperature"] == 0
    parts = body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["image_url", "text"]
    assert parts[0]["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()
    assert parts[1]["text"] == "Which?"


def test_sequence_interleaves_text_between_images():
    """A text part between images stops llama.cpp merging consecutive images
    into one video input."""
    rec = Recorder()
    client = OpenAICompatibleClient(MODELS, transport=rec.transport())
    client(job("vlm", payload("image_sequence", 3)))
    parts = json.loads(rec.requests[0].content)["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"] * 3 + ["text"]
    assert parts[0]["text"] == "Frame 1 (+0.0s):"
    assert parts[2]["text"] == "Frame 2 (+2.0s):"
    assert parts[4]["text"] == "Frame 3 (+4.0s):"
    assert parts[-1]["text"] == "Which?"


def test_timeout_comes_from_the_step():
    rec = Recorder()
    client = OpenAICompatibleClient(MODELS, transport=rec.transport())
    client(job("vlm", payload("image", 1, timeout_s=7.5)))
    assert rec.requests[0].extensions["timeout"]["read"] == 7.5


def test_bearer_from_named_env_var(monkeypatch):
    models = {"vlm": MODELS["vlm"].model_copy(update={"api_key_env": "PROXY_KEY"})}
    monkeypatch.setenv("PROXY_KEY", "sk-test")
    rec = Recorder()
    OpenAICompatibleClient(models, transport=rec.transport())(job("vlm", payload("image", 1)))
    assert rec.requests[0].headers["authorization"] == "Bearer sk-test"


def test_endpoint_trailing_slash_and_routing_by_model():
    rec = Recorder()
    client = OpenAICompatibleClient(MODELS, transport=rec.transport())
    with pytest.raises(VlmClientError, match="video_clip"):
        client(job("omni", payload("video_clip", 2)))
    assert rec.requests == []  # rejected before any request
    with pytest.raises(VlmClientError, match="unknown model"):
        client(job("ghost", payload("image", 1)))


def test_frames_without_data_are_rejected():
    p = StepPayload(kind="image", prompt="?", frames=(Frame(camera="cam", ts=1.0),), camera="cam",
                    chain_id="c", step_id="s", generation=1, center_ts=1.0)
    with pytest.raises(VlmClientError, match="no image data"):
        OpenAICompatibleClient(MODELS, transport=Recorder().transport())(job("vlm", p))


def test_http_error_and_malformed_body_become_client_errors():
    with pytest.raises(VlmClientError, match="HTTP 503"):
        OpenAICompatibleClient(MODELS, transport=Recorder(status=503).transport())(job("vlm", payload("image", 1)))
    with pytest.raises(VlmClientError, match="malformed"):
        OpenAICompatibleClient(MODELS, transport=Recorder(raw=b"not json").transport())(job("vlm", payload("image", 1)))
