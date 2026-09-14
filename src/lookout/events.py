"""Core event types and the tier-1 source seam.

Everything enters the engine as a DetectionEvent, regardless of what produced it.
That seam is what lets tier-1 be swapped without touching the engine:

  - OpenVinoVideoSource — v1: Ultralytics YOLO (OpenVINO export) over an OpenCV
    VideoCapture URI (file path or RTSP), with label debounce.
  - ReplaySource — test harness: yields canned events from a fixture file, so the
    engine, scheduler, and chains are fully exercisable with no camera, no GPU,
    and no network.
  - FrigateMqttSource — future: consume Frigate's tracked-object events over MQTT
    instead of running our own detector. Deliberately unimplemented in v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass(frozen=True)
class DetectionEvent:
    """One debounced tier-1 detection.

    frame_ref is an opaque reference (in v1: a key into the camera's frame ring
    buffer) that escalation payload builders resolve into an image, an image
    sequence, or a clip. The event itself never carries pixel data.
    """

    camera: str
    ts: float
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    frame_ref: str


class Source(Protocol):
    """A tier-1 event producer. Implementations are generators; the engine just
    iterates. Blocking inside the iterator is fine — each source runs in its own
    thread."""

    def events(self) -> Iterator[DetectionEvent]: ...
