"""Tier-1 from Frigate: its tracked objects, over MQTT, become DetectionEvents.

Frigate already runs a detector and an object tracker on every camera, and
records. Reading its tracker instead of running our own detector removes the
two things lookout's own tier-1 handled worst (parked cars re-firing on
detection flicker, passing traffic firing at all), and the pod no longer
decodes a video stream.

Frigate publishes each tracked object to `<prefix>/events` as it changes:
{"type": "new" | "update" | "end", "before": {...}, "after": {...}}, where
`after` carries id, camera, label, score, box, stationary, current_zones,
false_positive and more (frigate/track/tracked_object.py `to_dict`).
ArrivalGate turns that into one arrival per tracked object; see
FrigateCameraSpec for the rule.

The step's image is Frigate's snapshot of that object, fetched over HTTP when
the arrival fires and yielded as a Frame just before its event, so the
engine's image rule (the frame nearest the trigger) picks exactly it. Only
`image` steps work on a Frigate camera; config load says so.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from lookout.config import FrigateCameraSpec, FrigateSpec
from lookout.events import DetectionEvent, Frame

log = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Arrival:
    object_id: str
    label: str
    score: float
    box: BBox
    travel: float  # in first-box sizes


def _centre(box: BBox) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class ArrivalGate:
    """One arrival per tracked object: the first message in which it is a real
    detection (not Frigate's false_positive), stationary, in one of the zones,
    and has travelled at least `min_travel` first-box sizes since lookout first
    saw it. Objects are forgotten on Frigate's `end` message."""

    def __init__(self, spec: FrigateCameraSpec) -> None:
        self.spec = spec
        self._first_box: dict[str, BBox] = {}
        self._fired: set[str] = set()

    def update(self, message: dict[str, Any]) -> Arrival | None:
        obj = message.get("after") or {}
        object_id = obj.get("id")
        if obj.get("camera") != self.spec.camera or not object_id:
            return None
        if message.get("type") == "end":
            self._first_box.pop(object_id, None)
            self._fired.discard(object_id)
            return None
        box = tuple(int(v) for v in obj.get("box") or (0, 0, 0, 0))
        first = self._first_box.setdefault(object_id, box)  # type: ignore[arg-type]
        if object_id in self._fired or obj.get("false_positive", True) or not obj.get("stationary"):
            return None
        if not set(self.spec.zones) & set(obj.get("current_zones") or ()):
            return None
        (x0, y0), (x1, y1) = _centre(first), _centre(box)  # type: ignore[arg-type]
        size = max(first[2] - first[0], first[3] - first[1], 1)
        travel = math.hypot(x1 - x0, y1 - y0) / size
        if travel < self.spec.min_travel:
            return None
        self._fired.add(object_id)
        return Arrival(
            object_id=object_id, label=str(obj.get("label")), score=float(obj.get("score") or 0.0),
            box=box, travel=travel,  # type: ignore[arg-type]
        )


class FrigateError(RuntimeError):
    pass


class FrigateSnapshots:
    """Fetches an object's snapshot from Frigate's HTTP API. For an object
    still being tracked, Frigate serves its best frame so far."""

    def __init__(self, spec: FrigateSpec, transport: httpx.BaseTransport | None = None) -> None:
        self.url = spec.url.rstrip("/")
        self.timeout_s = spec.timeout_s
        self._http = httpx.Client(transport=transport) if transport else httpx.Client()

    def close(self) -> None:
        self._http.close()

    def snapshot(self, object_id: str, crop: bool, height: int) -> bytes:
        url = f"{self.url}/api/events/{object_id}/snapshot.jpg"
        params = {"crop": int(crop), "bbox": 0, "h": height, "quality": 90}
        try:
            response = self._http.get(url, params=params, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise FrigateError(f"snapshot of {object_id}: {exc}") from exc
        if response.status_code != 200 or not response.content:
            raise FrigateError(f"snapshot of {object_id}: HTTP {response.status_code}")
        return response.content


def mqtt_messages(spec: FrigateSpec, client_id: str) -> Iterator[dict[str, Any]]:
    """Frigate's event messages, decoded, for as long as the process runs.
    paho's network thread reconnects on its own and resubscribes in
    on_connect; a broker that refuses the first connection raises, and the
    process ends (the kubelet's restart is the retry, as for a dead camera)."""
    import paho.mqtt.client as mqtt

    mqtt_spec = spec.mqtt
    topic = f"{mqtt_spec.topic_prefix}/events"
    inbox: queue.Queue[bytes] = queue.Queue()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    if mqtt_spec.username:
        client.username_pw_set(mqtt_spec.username, mqtt_spec.password())

    def on_connect(client: Any, userdata: Any, flags: Any, reason: Any, properties: Any) -> None:
        if reason.is_failure:
            log.error("mqtt %s:%s refused the connection: %s", mqtt_spec.host, mqtt_spec.port, reason)
            return
        log.info("mqtt connected to %s:%s, subscribing to %s", mqtt_spec.host, mqtt_spec.port, topic)
        client.subscribe(topic)

    def on_disconnect(client: Any, userdata: Any, flags: Any, reason: Any, properties: Any) -> None:
        log.warning("mqtt disconnected (%s); paho will reconnect", reason)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = lambda client, userdata, msg: inbox.put(msg.payload)
    client.connect(mqtt_spec.host, mqtt_spec.port)
    client.loop_start()
    try:
        while True:
            try:
                raw = inbox.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                yield json.loads(raw)
            except ValueError:
                log.warning("mqtt: dropped a non-JSON message on %s", topic)
    finally:
        client.loop_stop()
        client.disconnect()


class FrigateSource:
    """A Source for one lookout camera fed by one Frigate camera.

    `messages` is the decoded event stream; the default is a live MQTT
    subscription, and tests pass a list. Timestamps are seconds since the
    stream started on `clock`, like VideoSource's for a live camera."""

    def __init__(
        self,
        camera: str,
        spec: FrigateCameraSpec,
        snapshots: FrigateSnapshots,
        messages: Iterable[dict[str, Any]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera = camera
        self.spec = spec
        self.snapshots = snapshots
        self.messages = messages
        self.clock = clock
        self.gate = ArrivalGate(spec)
        self.messages_read = 0

    def stream(self) -> Iterator[Frame | DetectionEvent]:
        t0 = self.clock()
        for message in self.messages:
            self.messages_read += 1
            arrival = self.gate.update(message)
            if arrival is None:
                continue
            ts = self.clock() - t0
            ref = f"frigate:{arrival.object_id}"
            try:
                jpeg = self.snapshots.snapshot(arrival.object_id, self.spec.crop, self.spec.snapshot_height)
            except FrigateError as exc:
                # No image means the step would have nothing to look at.
                log.error("%s: %s arrived but %s; skipping it", self.camera, arrival.label, exc)
                continue
            log.info(
                "%s: frigate %s %s stationary in %s, travelled %.1f box sizes",
                self.camera, arrival.label, arrival.object_id, "/".join(self.spec.zones), arrival.travel,
            )
            yield Frame(camera=self.camera, ts=ts, data=jpeg, ref=ref)
            yield DetectionEvent(
                camera=self.camera, ts=ts, label=arrival.label, confidence=arrival.score,
                bbox=arrival.box, frame_ref=ref,
            )
