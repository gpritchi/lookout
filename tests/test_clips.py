"""MediaMTX clip provider: URL shape, wall-clock conversion, retries. Plus the
engine's video_clip dispatch and the client's input_video part, with fakes."""

import base64
import json

import httpx
import pytest

from lookout.actions import MockSink
from lookout.clips import ClipError, MediaMtxClips
from lookout.config import Config, ConfigError, ModelSpec
from lookout.engine import Engine, StepPayload
from lookout.events import DetectionEvent, Frame
from lookout.frames import FrameBuffer
from lookout.runtime import VirtualClock
from lookout.scheduler import InferenceJob, Scheduler
from lookout.vlm import OpenAICompatibleClient

MP4 = b"\x00\x00\x00\x18ftypmp42fake"


class Playback:
    def __init__(self, fail_first: int = 0):
        self.requests: list[httpx.Request] = []
        self.fail_first = fail_first

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if len(self.requests) <= self.fail_first:
                return httpx.Response(404, text="segment not found")
            return httpx.Response(200, content=MP4)
        return httpx.MockTransport(handle)


def test_clip_request_is_wall_clock_and_duration():
    pb = Playback()
    clips = MediaMtxClips({"driveway": ("http://rec:9996", "drv")}, epoch=1_700_000_000.0,
                          transport=pb.transport(), retry_s=0)
    assert clips.clip("driveway", 10.0, 22.5) == MP4
    req = pb.requests[0]
    assert req.url.path == "/get"
    assert req.url.params["path"] == "drv"
    assert req.url.params["start"] == "2023-11-14T22:13:30.000Z"  # epoch + 10 s
    assert req.url.params["duration"] == "12.500"
    assert req.url.params["format"] == "mp4"


def test_clip_retries_then_succeeds_or_raises():
    pb = Playback(fail_first=2)
    clips = MediaMtxClips({"cam": ("http://rec:9996", "cam")}, epoch=0.0, transport=pb.transport(), retry_s=0)
    assert clips.clip("cam", 0.0, 5.0) == MP4
    assert len(pb.requests) == 3
    pb = Playback(fail_first=99)
    clips = MediaMtxClips({"cam": ("http://rec:9996", "cam")}, epoch=0.0, transport=pb.transport(), retry_s=0, attempts=2)
    with pytest.raises(ClipError, match="after 2 attempts"):
        clips.clip("cam", 0.0, 5.0)
    with pytest.raises(ClipError, match="no clip source"):
        clips.clip("other", 0.0, 5.0)


CONFIG = {
    "models": {"omni": {"endpoint": "http://proxy/v1", "model": "nemotron", "capabilities": ["video_clip"]}},
    "cameras": {"driveway": {"source": "rtsp://cam/x", "clips": {"playback": "http://rec:9996", "path": "driveway"}}},
    "chains": [{
        "id": "porch", "camera": "driveway", "trigger": {"label": "person"},
        "steps": [{
            "id": "watch", "model": "omni", "payload": "video_clip", "timeout_s": 60, "prompt": "Delivered or taken?",
            "window": {"before_s": 5, "after_s": 10},
            "outcomes": {"DELIVERED": {"end": True}, "TAKEN": {"action": {"type": "notify", "message": "theft"}}},
        }],
    }],
}


def test_video_clip_step_requires_a_camera_with_clips():
    data = json.loads(json.dumps(CONFIG))
    del data["cameras"]["driveway"]["clips"]
    with pytest.raises(ConfigError, match="no `clips` source"):
        Config.from_dict(data)
    Config.from_dict(CONFIG)


class FakeClips:
    def __init__(self):
        self.calls = []

    def clip(self, camera, start_ts, end_ts):
        self.calls.append((camera, start_ts, end_ts))
        return MP4


def test_engine_dispatches_clip_after_window_and_payload_carries_it():
    config = Config.from_dict(CONFIG)
    clock, scheduler, sink, clips = VirtualClock(), Scheduler(), MockSink(), FakeClips()
    trace = []
    engine = Engine(config, scheduler, sink, {"driveway": FrameBuffer(keep_s=60)}, clock.now, trace=trace.append, clips=clips)
    clock.advance_to(20.0)
    engine.on_event(DetectionEvent(camera="driveway", ts=20.0, label="person", confidence=0.9, bbox=(0, 0, 1, 1), frame_ref="r"))
    assert scheduler.pending() == 0 and engine.next_wakeup() == 30.0  # waits for after_s
    clock.advance_to(30.0)
    engine.tick()
    job = scheduler.next_job(["omni"], timeout=0)
    assert clips.calls == [("driveway", 15.0, 30.0)]
    assert job.payload.kind == "video_clip" and job.payload.clip == MP4 and job.payload.frames == ()
    assert any("clip 15.00..30.00" in line for line in trace)
    engine.on_result(job, "TAKEN")
    assert [a.message for a, _ in sink.fired] == ["theft"]


def test_engine_resets_chain_when_clip_unavailable():
    config = Config.from_dict(CONFIG)
    clock, scheduler = VirtualClock(), Scheduler()
    trace = []

    class Broken:
        def clip(self, *a):
            raise ClipError("recorder down")

    engine = Engine(config, scheduler, MockSink(), {"driveway": FrameBuffer(keep_s=60)}, clock.now, trace=trace.append, clips=Broken())
    clock.advance_to(20.0)
    engine.on_event(DetectionEvent(camera="driveway", ts=20.0, label="person", confidence=0.9, bbox=(0, 0, 1, 1), frame_ref="r"))
    clock.advance_to(30.0)
    engine.tick()
    assert engine.active == [] and scheduler.pending() == 0
    assert any("clip failed (recorder down)" in line for line in trace)


def test_client_sends_input_video_part():
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "TAKEN"}}]})

    client = OpenAICompatibleClient(
        {"omni": ModelSpec(endpoint="http://proxy/v1", model="nemotron", capabilities=["video_clip"])},
        transport=httpx.MockTransport(handle),
    )
    payload = StepPayload(kind="video_clip", prompt="Delivered or taken?", frames=(), camera="driveway",
                          chain_id="porch", step_id="watch", generation=1, center_ts=20.0, clip=MP4)
    assert client(InferenceJob(model="omni", priority=1, chain_id="porch", step_id="watch", payload=payload)) == "TAKEN"
    parts = seen[0]["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["input_video", "text"]
    assert parts[0]["input_video"]["data"] == base64.b64encode(MP4).decode()  # raw base64, no data: prefix
    assert seen[0]["model"] == "nemotron"
