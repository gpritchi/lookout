"""Core event types and the tier-1 source seam.

Everything enters the engine as a DetectionEvent, regardless of what produced it.
That seam is what lets tier-1 be swapped without touching the engine:

  - OpenVinoVideoSource — v1: Ultralytics YOLO (OpenVINO export) over an OpenCV
    VideoCapture URI (file path or RTSP), with label debounce.
  - ReplaySource — test harness and offline demo: yields canned events from a
    JSONL fixture, plus synthetic frames so windows have something to sample,
    so the engine, scheduler, and chains are fully exercisable with no camera,
    no GPU, and no network.
  - FrigateMqttSource — future: consume Frigate's tracked-object events over MQTT
    instead of running our own detector. Deliberately unimplemented in v1.

A source yields Frames as well as events, in time order. Frames feed the
per-camera ring buffer that escalation payloads are sliced from; events say
"something worth a chain start happened at this timestamp".
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Frame:
    """One frame of one camera's stream. `data` is whatever the source produces
    (a numpy array from OpenCV, or None for replay fixtures); `ref` is a short
    human-readable id for logs."""

    camera: str
    ts: float
    data: Any = None
    ref: str = ""


@dataclass(frozen=True)
class DetectionEvent:
    """One debounced tier-1 detection.

    frame_ref names the frame the detector fired on. Escalation payloads are
    sliced from the camera's frame buffer around `ts`, so the event itself never
    carries pixel data.
    """

    camera: str
    ts: float
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    frame_ref: str


class Source(Protocol):
    """A tier-1 producer. Implementations are generators; the engine's runner
    just iterates. Blocking inside the iterator is fine — each source runs in
    its own thread in a live deployment."""

    def stream(self) -> Iterator[Frame | DetectionEvent]: ...


class ReplaySource:
    """Replays a JSONL fixture of detection events, interleaved with synthetic
    frames at `fps` so window sampling has timestamps to pick from.

    Fixture lines look like:
        {"at_s": 3.0, "camera": "driveway", "label": "car",
         "confidence": 0.91, "bbox": [412, 220, 880, 610]}

    `at_s` is seconds from replay start; the stream's time base starts at 0.
    Frames run from 0 to the last event plus `tail_s`, per camera seen in the
    fixture, so steps with an after_s window still have footage to look at.
    Pacing (real time vs as-fast-as-possible) is the runner's job, not this
    class's: it just yields in timestamp order.
    """

    def __init__(self, path: str | Path, fps: float = 2.0, tail_s: float = 30.0) -> None:
        self.path = Path(path)
        self.fps = fps
        self.tail_s = tail_s

    def _load_events(self) -> list[DetectionEvent]:
        events: list[DetectionEvent] = []
        with self.path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                row = json.loads(line)
                try:
                    events.append(
                        DetectionEvent(
                            camera=row["camera"],
                            ts=float(row["at_s"]),
                            label=row["label"],
                            confidence=float(row.get("confidence", 1.0)),
                            bbox=tuple(row.get("bbox", (0, 0, 0, 0))),
                            frame_ref=f"{row['camera']}@{float(row['at_s']):.2f}",
                        )
                    )
                except KeyError as exc:
                    raise ValueError(f"{self.path}:{lineno}: missing field {exc}") from exc
        events.sort(key=lambda e: e.ts)
        return events

    def stream(self) -> Iterator[Frame | DetectionEvent]:
        events = self._load_events()
        cameras = sorted({e.camera for e in events})
        end = (events[-1].ts if events else 0.0) + self.tail_s
        step = 1.0 / self.fps
        # Frames first at equal timestamps, so an event's own frame is already
        # in the buffer when the engine sees the event.
        pending = list(events)
        n = 0
        while True:
            ts = round(n * step, 6)
            if ts > end:
                break
            while pending and pending[0].ts < ts:
                yield pending.pop(0)
            for camera in cameras:
                yield Frame(camera=camera, ts=ts, data=None, ref=f"{camera}@{ts:.2f}")
            while pending and pending[0].ts == ts:
                yield pending.pop(0)
            n += 1
        yield from pending
