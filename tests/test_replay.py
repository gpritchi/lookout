"""ReplaySource and the offline runner: the shipped fixture produces exactly
the actions the answer script implies."""

from pathlib import Path

from lookout.config import load_config
from lookout.events import DetectionEvent, Frame, ReplaySource
from lookout.runtime import ScriptedModel, run_replay

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "config" / "chains.example.json"
EVENTS = ROOT / "fixtures" / "sample" / "driveway-replay.jsonl"
ANSWERS = ROOT / "fixtures" / "sample" / "fake-vlm-answers.json"


def test_replay_source_interleaves_frames_and_events_in_order():
    items = list(ReplaySource(EVENTS, fps=2.0, tail_s=1.0).stream())
    stamps = [i.ts for i in items]
    assert stamps == sorted(stamps)
    events = [i for i in items if isinstance(i, DetectionEvent)]
    assert [e.label for e in events] == ["car", "car", "person", "gesture:three_fingers", "car"]
    frames = [i for i in items if isinstance(i, Frame)]
    assert {f.camera for f in frames} == {"driveway", "living-room"}
    # the frame at an event's timestamp comes before the event
    first_car = items.index(events[0])
    assert any(isinstance(i, Frame) and i.ts == 3.0 and i.camera == "driveway" for i in items[:first_car])


def test_scripted_model_consumes_in_order_and_repeats_last():
    m = ScriptedModel({"s": ["A", "B"]})
    from lookout.scheduler import InferenceJob
    from lookout.engine import StepPayload
    job = InferenceJob(model="m", priority=1, chain_id="c", step_id="s",
                       payload=StepPayload("image", "", (), "cam", "c", "s", 1, 0.0))
    assert [m(job), m(job), m(job)] == ["A", "B", "B"]
    other = InferenceJob(model="m", priority=1, chain_id="c", step_id="zzz", payload=job.payload)
    assert m(other) == "UNSCRIPTED"


def test_shipped_fixture_end_to_end():
    config = load_config(CONFIG)
    model = ScriptedModel.from_file(ANSWERS)
    report = run_replay(config, ReplaySource(EVENTS), model)
    assert report.actions == [
        "ha_webhook(service=tv_netflix_on) <- tv-netflix on living-room",
        "notify(message=Package delivered) <- driveway-arrivals on driveway",
        "notify(message=Brother and sister-in-law are at the door) <- driveway-arrivals on driveway",
    ]
    assert report.inference_calls == 7
    kinds = [(p.step_id, p.kind, len(p.frames)) for p in model.calls]
    assert kinds == [
        ("classify-vehicle", "image", 1),
        ("check-driver", "image_sequence", 6),
        ("check-driver", "image_sequence", 6),
        ("confirm-delivery", "video_clip", 0),
        ("classify-vehicle", "image", 1),
        ("check-tacoma-driver", "image_sequence", 8),
        ("count-visitors", "video_clip", 0),
    ]
    # video steps get a placeholder clip in replay and wait out their after_s
    assert model.calls[3].clip == b"replay-placeholder"
    assert any("confirm-delivery: -> nemotron-omni p40 video_clip x0 [clip -2.00..33.00" in line for line in report.trace)
    # the first check-driver window is [1, 11] around the 3.0 trigger; the retry re-centres to end at 16
    assert (model.calls[1].frames[0].ts, model.calls[1].frames[-1].ts) == (1.0, 11.0)
    assert (model.calls[2].frames[0].ts, model.calls[2].frames[-1].ts) == (6.0, 16.0)
