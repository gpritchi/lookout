"""Tier-1: the cheap detector over a video stream.

YoloDetector wraps an Ultralytics model (OpenVINO export by default, PyTorch
weights as the fallback) and returns COCO-class detections for one frame. Every
call is timed under {model, tier="tier1"} so the write-up's "milliseconds vs
seconds" claim has numbers.

ArrivalDebouncer turns per-frame detections into events. The naive rule, "label
present for N frames", is wrong for a driveway: a parked car is present in every
frame, so it would fire at second zero and never again. What a chain wants is
"a NEW car appeared". Without tracking, the cheapest honest proxy is the count
of a label rising above its baseline and staying there for N analysed frames.
The baseline follows the count back down after N frames below it, so a car
leaving re-arms the trigger. The event's bbox is the detection that overlaps the
baseline set least, i.e. the newcomer.

VideoSource reads an OpenCV URI (file path, RTSP URL, or device index), feeds
every `frame_fps`-th frame into the ring buffer as a JPEG, runs the detector at
`analysis_fps`, and yields events from the debouncer. For files the time base
is the video's own clock; for live sources it is the wall clock.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lookout.config import Tier1Spec
from lookout.events import DetectionEvent, Frame
from lookout.metrics import timed

log = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox


Detector = Callable[[np.ndarray], list[Detection]]


class YoloDetector:
    """Ultralytics YOLO with the OpenVINO export, exporting on first use if the
    directory is missing. `backend="torch"` uses the .pt weights directly."""

    def __init__(self, spec: Tier1Spec, models_dir: str | Path = ".") -> None:
        from ultralytics import YOLO  # slow import; keep it out of module load

        self.spec = spec
        models_dir = Path(models_dir)
        weights = models_dir / f"{spec.model}.pt"
        if spec.backend == "openvino":
            exported = models_dir / f"{spec.model}_openvino_model"
            if not exported.exists():
                log.info("exporting %s to OpenVINO at %s (one-time)", spec.model, exported)
                YOLO(str(weights)).export(format="openvino", imgsz=spec.imgsz, half=False)
            self._model = YOLO(str(exported), task="detect")
        else:
            self._model = YOLO(str(weights))
        self.name = f"{spec.model}-{spec.backend}"
        # Predict down to count_confidence; the debouncer applies min_confidence
        # to the newcomer only. See Tier1Spec.
        self._predict_kwargs = {"imgsz": spec.imgsz, "conf": spec.count_confidence, "verbose": False}
        if spec.device:
            self._predict_kwargs["device"] = spec.device

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        with timed(self.name, "tier1"):
            result = self._model.predict(frame_bgr, **self._predict_kwargs)[0]
        h, w = frame_bgr.shape[:2]
        min_area = self.spec.min_box_frac * w * h
        out: list[Detection] = []
        for cls, conf, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
            x1, y1, x2, y2 = (int(v) for v in xyxy.tolist())
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            out.append(Detection(result.names[int(cls)], float(conf), (x1, y1, x2, y2)))
        return out


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area else 0.0


@dataclass
class _LabelState:
    baseline: int
    baseline_boxes: list[BBox]
    above: int = 0
    below: int = 0


@dataclass
class ArrivalDebouncer:
    """Count-rise hysteresis per label. See the module docstring."""

    n_frames: int
    fire_confidence: float = 0.0  # the newcomer must reach this to raise an event
    # A count rise whose newcomer never gets confident (a ghost at 0.35 for
    # ages) is absorbed into the baseline after this many frames so it cannot
    # block later arrivals. Long, because a real arrival approaching from far
    # away spends a while below fire_confidence and must not be absorbed.
    absorb_after: int = 40
    _state: dict[str, _LabelState] = field(default_factory=dict)
    _frames_seen: int = 0

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Feed one analysed frame. Returns the detections to raise as events
        (at most one per label per frame)."""
        by_label: dict[str, list[Detection]] = {}
        for det in detections:
            by_label.setdefault(det.label, []).append(det)
        fired: list[Detection] = []
        first_frame = self._frames_seen == 0
        self._frames_seen += 1
        for label in sorted(set(by_label) | set(self._state)):
            dets = by_label.get(label, [])
            count = len(dets)
            state = self._state.get(label)
            if state is None:
                # On the very first frame, whatever is there is the baseline:
                # nothing fires for what was already present when we started
                # looking. A label first seen later was absent before, so its
                # baseline is 0 and this frame starts its streak.
                if first_frame:
                    self._state[label] = _LabelState(baseline=count, baseline_boxes=[d.bbox for d in dets])
                    continue
                state = self._state[label] = _LabelState(baseline=0, baseline_boxes=[])
            if count > state.baseline:
                state.above += 1
                state.below = 0
                if state.above >= self.n_frames:
                    newcomer = self._newcomer(dets, state.baseline_boxes)
                    if newcomer.confidence >= self.fire_confidence:
                        fired.append(newcomer)
                    elif state.above < self.absorb_after:
                        continue  # something new but not yet convincing: keep watching
                    state.baseline, state.baseline_boxes, state.above = count, [d.bbox for d in dets], 0
            elif count < state.baseline:
                state.below += 1
                state.above = 0
                if state.below >= self.n_frames:
                    state.baseline, state.baseline_boxes, state.below = count, [d.bbox for d in dets], 0
            else:
                state.above = state.below = 0
        return fired

    @staticmethod
    def _newcomer(dets: list[Detection], baseline_boxes: list[BBox]) -> Detection:
        """The most confident detection that does not overlap the baseline set.
        Falls back to the least-overlapping one if everything overlaps."""
        if not baseline_boxes:
            return max(dets, key=lambda d: d.confidence)
        overlap = {id(d): max(iou(d.bbox, b) for b in baseline_boxes) for d in dets}
        fresh = [d for d in dets if overlap[id(d)] < 0.5]
        if fresh:
            return max(fresh, key=lambda d: d.confidence)
        return min(dets, key=lambda d: overlap[id(d)])


def _is_live(uri: str) -> bool:
    return uri.isdigit() or "://" in uri


def encode_jpeg(frame_bgr: np.ndarray, max_width: int, quality: int) -> bytes:
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


class VideoSource:
    def __init__(
        self,
        camera: str,
        uri: str,
        detector: Detector,
        spec: Tier1Spec,
        loop: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera = camera
        self.uri = uri
        self.detector = detector
        self.spec = spec
        self.loop = loop
        self.clock = clock
        self.debouncer = ArrivalDebouncer(spec.debounce_frames, fire_confidence=spec.min_confidence)
        self.frames_read = 0
        self.frames_analysed = 0

    def stream(self) -> Iterator[Frame | DetectionEvent]:
        offset = 0.0
        while True:
            cap = cv2.VideoCapture(int(self.uri) if self.uri.isdigit() else self.uri)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video source {self.uri!r}")
            live = _is_live(self.uri)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            frame_stride = max(1, round(src_fps / self.spec.frame_fps))
            analysis_stride = max(1, round(src_fps / self.spec.analysis_fps))
            index = 0
            t0 = self.clock()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    ts = (self.clock() - t0) if live else offset + index / src_fps
                    self.frames_read += 1
                    if index % frame_stride == 0:
                        yield Frame(
                            camera=self.camera, ts=ts,
                            data=encode_jpeg(frame, self.spec.frame_max_width, self.spec.jpeg_quality),
                            ref=f"{self.camera}@{ts:.2f}",
                        )
                    if index % analysis_stride == 0:
                        self.frames_analysed += 1
                        for det in self.debouncer.update(self.detector(frame)):
                            yield DetectionEvent(
                                camera=self.camera, ts=ts, label=det.label, confidence=det.confidence,
                                bbox=det.bbox, frame_ref=f"{self.camera}@{ts:.2f}",
                            )
                    index += 1
            finally:
                cap.release()
            if not self.loop or live:
                return
            offset += index / src_fps
