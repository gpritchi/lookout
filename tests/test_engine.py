"""Chain engine state machine, driven by hand on a virtual clock. The scheduler
is real; the worker is the test, which pops jobs and feeds answers back."""

import pytest

from lookout.actions import MockSink
from lookout.config import Config
from lookout.engine import Engine, StepPayload, match_outcome
from lookout.events import DetectionEvent, Frame
from lookout.frames import FrameBuffer
from lookout.runtime import VirtualClock
from lookout.scheduler import Scheduler

CONFIG = {
    "models": {"vlm": {"endpoint": "http://localhost:4000/v1", "model": "m", "capabilities": ["image", "image_sequence"]}},
    "cameras": {"driveway": {"source": "x.mp4"}, "living": {"source": "y.mp4"}},
    "chains": [
        {
            "id": "arrivals",
            "camera": "driveway",
            "trigger": {"label": "car", "priority": 10},
            "window": {"before_s": 2, "after_s": 8, "frames": 6},
            "steps": [
                {
                    "id": "classify", "model": "vlm", "payload": "image", "priority": 50, "timeout_s": 30,
                    "prompt": "which?",
                    "outcomes": {"VAN": {"next": "driver"}, "OTHER": {"end": True}},
                },
                {
                    "id": "driver", "model": "vlm", "payload": "image_sequence", "priority": 60, "timeout_s": 60,
                    "prompt": "who?",
                    "outcomes": {
                        "DELIVERY": {"action": {"type": "notify", "message": "delivery"}},
                        "NOBODY_YET": {"retry_in_s": 5},
                    },
                },
            ],
        },
        {
            "id": "tv",
            "camera": "living",
            "trigger": {"label": "gesture:ok", "held_frames": 8, "priority": 100},
            "steps": [],
            "on_trigger": {"action": {"type": "ha_webhook", "service": "tv_off"}},
        },
    ],
}


class Rig:
    def __init__(self):
        self.config = Config.from_dict(CONFIG)
        self.clock = VirtualClock()
        self.scheduler = Scheduler()
        self.sink = MockSink()
        self.buffers = {name: FrameBuffer(keep_s=120) for name in self.config.cameras}
        self.trace: list[str] = []
        self.engine = Engine(self.config, self.scheduler, self.sink, self.buffers, self.clock.now, trace=self.trace.append)

    def frames(self, camera: str, upto: float, fps: float = 2.0) -> None:
        """Feed frames up to `upto` and move the clock there, ticking as we go."""
        ts = 0.0
        while ts <= upto + 1e-9:
            if self.buffers[camera].latest() is None or ts > self.buffers[camera].latest().ts:
                self.clock.advance_to(ts)
                self.engine.on_frame(Frame(camera=camera, ts=ts))
                self.engine.tick()
            ts = round(ts + 1 / fps, 6)
        self.clock.advance_to(upto)
        self.engine.tick()

    def event(self, camera: str, label: str, ts: float) -> None:
        self.clock.advance_to(ts)
        self.engine.on_event(DetectionEvent(camera=camera, ts=ts, label=label, confidence=0.9, bbox=(0, 0, 1, 1), frame_ref="r"))

    def pop(self):
        return self.scheduler.next_job(["vlm"], timeout=0)


def test_image_step_dispatches_immediately_with_trigger_frame():
    rig = Rig()
    rig.frames("driveway", 3.0)
    rig.event("driveway", "car", 3.0)
    job = rig.pop()
    assert job is not None and job.step_id == "classify" and job.priority == 50
    payload: StepPayload = job.payload
    assert payload.kind == "image" and [f.ts for f in payload.frames] == [3.0]
    assert payload.prompt == "which?"


def test_next_step_waits_for_after_s_then_samples_window():
    rig = Rig()
    rig.frames("driveway", 3.0)
    rig.event("driveway", "car", 3.0)
    job = rig.pop()
    rig.engine.on_result(job, "VAN")
    assert rig.pop() is None  # after_s=8 not yet elapsed
    assert rig.engine.next_wakeup() == 11.0
    rig.frames("driveway", 10.5)
    assert rig.pop() is None
    rig.frames("driveway", 11.0)
    job = rig.pop()
    assert job is not None and job.step_id == "driver" and job.priority == 60
    assert [f.ts for f in job.payload.frames] == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]


def test_action_outcome_fires_sink_and_ends_run():
    rig = Rig()
    rig.frames("driveway", 3.0)
    rig.event("driveway", "car", 3.0)
    rig.engine.on_result(rig.pop(), "VAN")
    rig.frames("driveway", 11.0)
    rig.engine.on_result(rig.pop(), "DELIVERY")
    assert [(a.type, a.message) for a, _ in rig.sink.fired] == [("notify", "delivery")]
    ctx = rig.sink.fired[0][1]
    assert (ctx.chain_id, ctx.step_id, ctx.answer, ctx.trigger_ts) == ("arrivals", "driver", "DELIVERY", 3.0)
    assert rig.engine.active == []


def test_end_outcome_ends_without_action():
    rig = Rig()
    rig.frames("driveway", 3.0)
    rig.event("driveway", "car", 3.0)
    rig.engine.on_result(rig.pop(), "OTHER")
    assert rig.sink.fired == [] and rig.engine.active == []


def test_retry_recenters_window_to_end_at_retry_time():
    rig = Rig()
    rig.frames("driveway", 3.0)
    rig.event("driveway", "car", 3.0)
    rig.engine.on_result(rig.pop(), "VAN")
    rig.frames("driveway", 11.0)
    rig.engine.on_result(rig.pop(), "NOBODY_YET")
    assert rig.pop() is None
    assert rig.engine.next_wakeup() == 16.0
    rig.frames("driveway", 16.0)
    job = rig.pop()
    assert job.step_id == "driver"
    assert [f.ts for f in job.payload.frames] == [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]


def test_timeout_resets_chain_drops_queued_job_and_ignores_late_answer():
    rig = Rig()
    rig.frames("driveway", 3.0)
    rig.event("driveway", "car", 3.0)
    assert rig.scheduler.pending("vlm") == 1  # queued, never picked up
    rig.frames("driveway", 33.5)  # past classify's timeout_s=30
    assert rig.engine.active == []
    assert rig.scheduler.pending("vlm") == 0
    assert any("timed out" in line for line in rig.trace)
    # a fresh trigger starts a new generation; an answer for the old one is ignored
    rig.event("driveway", "car", 34.0)
    new_job = rig.pop()
    stale = StepPayload(kind="image", prompt="", frames=(), camera="driveway", chain_id="arrivals",
                        step_id="classify", generation=new_job.payload.generation - 1, center_ts=3.0)
    from lookout.scheduler import InferenceJob
    rig.engine.on_result(InferenceJob(model="vlm", priority=1, chain_id="arrivals", step_id="classify", payload=stale), "VAN")
    assert rig.engine.active == ["arrivals"]
    assert rig.scheduler.pending("vlm") == 0  # stale answer did not advance anything


def test_retrigger_while_active_is_ignored():
    rig = Rig()
    rig.frames("driveway", 3.0)
    rig.event("driveway", "car", 3.0)
    rig.event("driveway", "car", 3.5)
    assert rig.scheduler.pending("vlm") == 1
    assert any("already active" in line for line in rig.trace)


def test_steps_less_chain_fires_on_trigger():
    rig = Rig()
    rig.event("living", "gesture:ok", 5.0)
    assert [(a.type, a.service) for a, _ in rig.sink.fired] == [("ha_webhook", "tv_off")]
    assert rig.scheduler.pending() == 0 and rig.engine.active == []


def test_unrecognised_answer_resets_chain():
    rig = Rig()
    rig.frames("driveway", 3.0)
    rig.event("driveway", "car", 3.0)
    rig.engine.on_result(rig.pop(), "I am not sure what that is")
    assert rig.engine.active == [] and rig.sink.fired == []
    assert any("unrecognised answer" in line for line in rig.trace)


def test_inference_error_resets_chain():
    rig = Rig()
    rig.frames("driveway", 3.0)
    rig.event("driveway", "car", 3.0)
    rig.engine.on_error(rig.pop(), RuntimeError("endpoint down"))
    assert rig.engine.active == []


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("VAN", "VAN"),
        ("van", "VAN"),
        ("  VAN \n", "VAN"),
        ("The vehicle is a grey VAN.", "VAN"),
        ("Answer: OTHER", "OTHER"),
        ("VANGUARD", None),  # not a whole word
        ("nothing useful", None),
    ],
)
def test_match_outcome(answer, expected):
    outcomes = Config.from_dict(CONFIG).chain("arrivals").steps[0].outcomes
    assert match_outcome(answer.strip(), outcomes) == expected
