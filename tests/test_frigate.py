"""Frigate as tier-1: the arrival rule, the source's frame-then-event order, and
the config rules for a Frigate camera. Messages are shaped like Frigate 0.18's
`frigate/events` payloads (tracked_object.py `to_dict`), trimmed to the fields
lookout reads."""

from copy import deepcopy

import httpx
import pytest

from lookout.config import Config, ConfigError, FrigateCameraSpec, FrigateSpec
from lookout.events import DetectionEvent, Frame
from lookout.frigate import ArrivalGate, FrigateSnapshots, FrigateSource

SPEC = FrigateCameraSpec(camera="outside", zones=["driveway"])


def msg(box, *, id="car-1", type="update", stationary=False, zones=(), camera="outside", label="car", fp=False):
    return {
        "type": type,
        "before": {},
        "after": {
            "id": id, "camera": camera, "label": label, "score": 0.8, "box": list(box),
            "stationary": stationary, "active": not stationary, "current_zones": list(zones),
            "false_positive": fp,
        },
    }


STREET = (100, 100, 200, 150)  # first seen out on the street, 100 px wide
PARKED = (400, 500, 600, 650)  # stopped in the driveway, well over one first-box size away


def arrival_messages(id="car-1"):
    return [
        msg(STREET, id=id, type="new"),
        msg((250, 300, 400, 420), id=id, zones=["driveway"]),
        msg(PARKED, id=id, zones=["driveway"], stationary=True),
    ]


def test_car_that_drives_in_and_stops_arrives_once():
    gate = ArrivalGate(SPEC)
    results = [gate.update(m) for m in arrival_messages()]
    assert results[:2] == [None, None]
    assert results[2] is not None and results[2].object_id == "car-1" and results[2].box == PARKED
    # Frigate keeps publishing the parked car; it does not arrive again.
    assert gate.update(msg(PARKED, zones=["driveway"], stationary=True)) is None


def test_car_parked_since_before_lookout_started_never_arrives():
    """First seen already parked (a Frigate or lookout restart): it has
    travelled nowhere, however long it sits in the zone."""
    gate = ArrivalGate(SPEC)
    for jitter in range(5):
        box = (PARKED[0] + jitter, PARKED[1], PARKED[2] + jitter, PARKED[3])
        assert gate.update(msg(box, zones=["driveway"], stationary=True)) is None


def test_passing_traffic_never_stops_in_the_zone():
    gate = ArrivalGate(SPEC)
    for x in range(0, 1200, 100):
        assert gate.update(msg((x, 100, x + 100, 150), zones=["driveway"] if 400 < x < 800 else [])) is None


def test_stopping_outside_the_zone_is_not_an_arrival():
    gate = ArrivalGate(SPEC)
    gate.update(msg(STREET, type="new"))
    assert gate.update(msg(PARKED, stationary=True, zones=["street"])) is None


def test_false_positive_and_other_cameras_are_ignored():
    gate = ArrivalGate(SPEC)
    gate.update(msg(STREET, type="new", fp=True))
    assert gate.update(msg(PARKED, zones=["driveway"], stationary=True, fp=True)) is None
    other = ArrivalGate(SPEC)
    assert all(other.update({**m, "after": {**m["after"], "camera": "garage"}}) is None for m in arrival_messages())


def test_end_forgets_the_object():
    gate = ArrivalGate(SPEC)
    for m in arrival_messages():
        gate.update(m)
    gate.update(msg(PARKED, type="end", zones=["driveway"], stationary=True))
    assert gate._first_box == {} and gate._fired == set()


FRIGATE = FrigateSpec.model_validate({"url": "http://frigate.test:5000", "mqtt": {"host": "broker.test"}})


def snapshots(status=200):
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, content=b"\xff\xd8jpeg" if status == 200 else b"")

    return FrigateSnapshots(FRIGATE, transport=httpx.MockTransport(handle)), seen


def test_source_yields_the_snapshot_frame_then_the_event_at_the_same_instant():
    snaps, seen = snapshots()
    source = FrigateSource("driveway", SPEC, snaps, arrival_messages(), clock=iter([0.0, 3.5]).__next__)
    frame, event = list(source.stream())
    assert isinstance(frame, Frame) and isinstance(event, DetectionEvent)
    assert frame.data == b"\xff\xd8jpeg" and frame.ts == event.ts == 3.5
    assert (event.camera, event.label, event.bbox, event.frame_ref) == ("driveway", "car", PARKED, "frigate:car-1")
    (request,) = seen
    assert request.url.path == "/api/events/car-1/snapshot.jpg"
    assert request.url.params["crop"] == "1" and request.url.params["bbox"] == "0"


def test_source_skips_an_arrival_whose_snapshot_fails():
    snaps, _ = snapshots(status=404)
    source = FrigateSource("driveway", SPEC, snaps, arrival_messages(), clock=lambda: 0.0)
    assert list(source.stream()) == []
    assert source.messages_read == 3


# -- config ---------------------------------------------------------------------


def frigate_config() -> dict:
    return {
        "models": {"vlm": {"endpoint": "http://localhost:4000/v1", "model": "vlm", "capabilities": ["image", "image_sequence"]}},
        "frigate": {"url": "http://frigate:5000", "mqtt": {"host": "broker", "username": "lookout", "password_env": "PW"}},
        "cameras": {"driveway": {"tier1": "frigate", "frigate": {"camera": "outside", "zones": ["driveway"]}}},
        "chains": [
            {
                "id": "arrivals", "camera": "driveway", "trigger": {"label": "car"},
                "steps": [{
                    "id": "classify", "model": "vlm", "payload": "image", "timeout_s": 30,
                    "prompt": "?", "outcomes": {"ANY": {"end": True}},
                }],
            }
        ],
    }


def test_frigate_camera_config_loads_without_a_source():
    config = Config.from_dict(frigate_config())
    assert config.cameras["driveway"].frigate.camera == "outside"
    assert config.cameras["driveway"].source is None


def test_frigate_camera_needs_the_top_level_frigate_block():
    data = frigate_config()
    del data["frigate"]
    with pytest.raises(ConfigError, match=r"camera 'driveway' reads Frigate.*no top-level `frigate`"):
        Config.from_dict(data)


def test_frigate_camera_needs_its_frigate_block():
    data = frigate_config()
    del data["cameras"]["driveway"]["frigate"]
    with pytest.raises(ConfigError, match=r"tier1 'frigate' needs a `frigate` block"):
        Config.from_dict(data)


def test_yolo_camera_still_needs_a_source():
    data = frigate_config()
    data["cameras"]["driveway"] = {"tier1": "yolo"}
    with pytest.raises(ConfigError, match=r"tier1 'yolo' needs a `source`"):
        Config.from_dict(data)


def test_frigate_camera_only_takes_image_steps():
    data = deepcopy(frigate_config())
    step = data["chains"][0]["steps"][0]
    step["payload"] = "image_sequence"
    step["window"] = {"before_s": 2, "after_s": 8, "frames": 6}
    with pytest.raises(ConfigError, match=r"step 'classify'.*reads Frigate.*only 'image' steps"):
        Config.from_dict(data)


def test_mqtt_password_is_read_from_the_environment(monkeypatch):
    config = Config.from_dict(frigate_config())
    monkeypatch.delenv("PW", raising=False)
    with pytest.raises(ConfigError, match=r"\$PW, which is not set"):
        config.frigate.mqtt.password()
    monkeypatch.setenv("PW", "secret")
    assert config.frigate.mqtt.password() == "secret"
