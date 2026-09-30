"""Tier-1: the cheap detector over a video stream.

OpenVinoDetector runs the YOLOv8 OpenVINO export with the openvino runtime
directly and returns COCO-class detections for one frame. It does its own pre-
and postprocessing (letterbox, NMS, box scaling), the same arithmetic as
Ultralytics' predictor, so neither Ultralytics nor torch is imported at runtime:
they are an install extra, needed only to export the model (once, on a laptop;
at build time, in the image) and for `backend: "torch"`, which runs the .pt
weights through Ultralytics. Every call is timed under {model, tier="tier1"} so
the write-up's "milliseconds vs seconds" claim has numbers.

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
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

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


class DetectorError(RuntimeError):
    """The detector cannot be loaded; the message says what to do about it."""


# Ultralytics' predict defaults, which every tuning run in this repo used.
NMS_IOU = 0.7
MAX_DET = 300
# Class-aware NMS in one pass: shift each class's boxes this far apart so boxes
# of different classes never overlap (Ultralytics' max_wh).
_CLASS_OFFSET = 7680.0
_PAD_VALUE = 114


def letterbox(frame_bgr: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Scale `frame_bgr` to fit `shape` (h, w) keeping its aspect ratio and pad
    the rest with grey, image centred. Returns the padded image, the scale
    gain, and the (left, top) padding. Rounding follows Ultralytics' LetterBox
    exactly (`round(d - 0.1)` puts the odd pixel on the bottom/right), because
    a one-pixel shift of the input shifts every box."""
    h0, w0 = frame_bgr.shape[:2]
    h, w = shape
    gain = min(h / h0, w / w0)
    new_w, new_h = round(w0 * gain), round(h0 * gain)
    dw, dh = (w - new_w) / 2, (h - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    if (new_w, new_h) != (w0, h0):
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        frame_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(_PAD_VALUE,) * 3
    )
    return padded, gain, (left, top)


def to_input(image_bgr: np.ndarray) -> np.ndarray:
    """HWC BGR uint8 -> 1x3xHxW RGB float32 in [0, 1]. Divides rather than
    multiplying by 1/255 so the floats match Ultralytics' bit for bit."""
    rgb = image_bgr[..., ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(rgb, dtype=np.float32) / np.float32(255)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS: indices of the kept boxes, highest score first. A box is
    dropped when its IoU with a kept box exceeds `iou_thres` (torchvision's rule,
    which Ultralytics uses)."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep: list[int] = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(int(i))
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[overlap <= iou_thres]
    return np.array(keep, dtype=np.intp)


def decode(
    pred: np.ndarray, conf_thres: float, iou_thres: float = NMS_IOU, max_det: int = MAX_DET
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YOLOv8 head output [4 + classes, anchors] (cx, cy, w, h, then one score
    per class) -> (xyxy boxes, scores, class ids) in model-input pixels, after
    the confidence threshold and class-aware NMS. Each anchor keeps only its
    best class, and must beat `conf_thres` strictly, as in Ultralytics."""
    rows = pred.T
    class_scores = rows[:, 4:]
    cls = class_scores.argmax(1)
    conf = class_scores[np.arange(len(rows)), cls]
    hit = conf > conf_thres
    xywh, conf, cls = rows[hit, :4], conf[hit], cls[hit]
    half = xywh[:, 2:] / 2
    boxes = np.concatenate([xywh[:, :2] - half, xywh[:, :2] + half], axis=1)
    offset = (cls.astype(np.float32) * np.float32(_CLASS_OFFSET))[:, None]
    keep = nms(boxes + offset, conf, iou_thres)[:max_det]
    return boxes[keep], conf[keep], cls[keep]


def unletterbox(boxes: np.ndarray, gain: float, pad: tuple[int, int], frame_shape: tuple[int, ...]) -> np.ndarray:
    """Map xyxy boxes from the letterboxed input back to the original frame,
    clipped to it."""
    h, w = frame_shape[:2]
    out = boxes.copy()
    out[:, [0, 2]] -= pad[0]
    out[:, [1, 3]] -= pad[1]
    out /= gain
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def _detections(
    boxes: np.ndarray, confs: np.ndarray, classes: np.ndarray, names: dict[int, str],
    frame_shape: tuple[int, ...], min_box_frac: float,
) -> list[Detection]:
    """Frame-pixel xyxy boxes -> Detections, integer boxes (truncated), dropping
    boxes smaller than `min_box_frac` of the frame."""
    h, w = frame_shape[:2]
    min_area = min_box_frac * w * h
    out: list[Detection] = []
    for cls, conf, xyxy in zip(classes, confs, boxes):
        x1, y1, x2, y2 = (int(v) for v in xyxy.tolist())
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        out.append(Detection(names[int(cls)], float(conf), (x1, y1, x2, y2)))
    return out


class OpenVinoDetector:
    """The exported YOLOv8 model on the openvino runtime. `exported` is the
    `<model>_openvino_model` directory Ultralytics writes: the IR (.xml/.bin)
    and a metadata.yaml with the class names."""

    def __init__(self, spec: Tier1Spec, exported: str | Path) -> None:
        import openvino as ov

        exported = Path(exported)
        xml = min(exported.glob("*.xml"), default=None)
        meta = exported / "metadata.yaml"
        if xml is None or not meta.is_file():
            raise DetectorError(f"{exported} is not a complete OpenVINO export (needs <model>.xml and metadata.yaml)")
        with meta.open() as fh:
            self.names: dict[int, str] = {int(k): v for k, v in yaml.safe_load(fh)["names"].items()}
        self.spec = spec
        self.name = f"{spec.model}-{spec.backend}"
        core = ov.Core()
        model = core.read_model(xml)
        shape = model.input(0).get_partial_shape()
        if shape.is_static:
            self.input_hw = (shape[2].get_length(), shape[3].get_length())
            if self.input_hw != (spec.imgsz, spec.imgsz):
                raise DetectorError(
                    f"{exported} was exported at {self.input_hw[0]}x{self.input_hw[1]} but tier1.imgsz is "
                    f"{spec.imgsz}; delete the export to re-export it at the configured size"
                )
        else:
            self.input_hw = (spec.imgsz, spec.imgsz)
        # Ultralytics' choice: CPU when that is all there is, else let AUTO pick.
        device = spec.device.upper() if spec.device else ("CPU" if core.available_devices == ["CPU"] else "AUTO")
        self._model = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
        self._output = self._model.output(0)
        # One compiled model serves every camera thread; its implicit infer
        # request is not safe to share, so calls take turns (Ultralytics'
        # predictor serialised them the same way).
        self._lock = threading.Lock()

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        with timed(self.name, "tier1"):
            image, gain, pad = letterbox(frame_bgr, self.input_hw)
            with self._lock:
                pred = self._model(to_input(image))[self._output][0]
            boxes, confs, classes = decode(pred, self.spec.count_confidence)
            boxes = unletterbox(boxes, gain, pad, frame_bgr.shape)
        return _detections(boxes, confs, classes, self.names, frame_bgr.shape, self.spec.min_box_frac)


def _import_yolo(why: str):  # -> ultralytics.YOLO, which may not be installed
    try:
        from ultralytics import YOLO  # slow import; only the export and torch paths need it
    except ImportError as exc:
        raise DetectorError(
            f"{why} needs Ultralytics and torch, which are an optional extra and not installed: "
            "`uv sync --extra torch` (or `pip install 'lookout[torch]'`)"
        ) from exc
    return YOLO


class UltralyticsDetector:
    """`backend: "torch"`: the .pt weights through Ultralytics. Needs the
    `torch` extra."""

    def __init__(self, spec: Tier1Spec, weights: str | Path) -> None:
        yolo = _import_yolo('tier1.backend "torch"')
        self.spec = spec
        self.name = f"{spec.model}-{spec.backend}"
        self._model = yolo(str(weights))
        # Predict down to count_confidence; the debouncer applies min_confidence
        # to the newcomer only. See Tier1Spec.
        self._predict_kwargs = {"imgsz": spec.imgsz, "conf": spec.count_confidence, "verbose": False}
        if spec.device:
            self._predict_kwargs["device"] = spec.device

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        with timed(self.name, "tier1"):
            result = self._model.predict(frame_bgr, **self._predict_kwargs)[0]
        boxes, confs, classes = (t.cpu().numpy() for t in (result.boxes.xyxy, result.boxes.conf, result.boxes.cls))
        return _detections(boxes, confs, classes, result.names, frame_bgr.shape, self.spec.min_box_frac)


def load_detector(spec: Tier1Spec, models_dir: str | Path = ".") -> OpenVinoDetector | UltralyticsDetector:
    """The configured detector. For OpenVINO, exports `<model>.pt` on first use
    if the export is missing, which needs the `torch` extra; the image ships
    the export, so it never does this."""
    models_dir = Path(models_dir)
    weights = models_dir / f"{spec.model}.pt"
    if spec.backend == "torch":
        return UltralyticsDetector(spec, weights)
    exported = models_dir / f"{spec.model}_openvino_model"
    if not exported.exists():
        yolo = _import_yolo(
            f"there is no OpenVINO export at {exported} (point --models-dir at one), and exporting {weights}"
        )
        log.info("exporting %s to OpenVINO at %s (one-time)", spec.model, exported)
        yolo(str(weights)).export(format="openvino", imgsz=spec.imgsz, half=False)
    return OpenVinoDetector(spec, exported)


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
class _Spot:
    """A place something of this group has been seen. The anchor is the first
    box and never moves, so a vehicle driving through leaves a trail of brief
    spots behind it, while a parked one piles up hits on a single spot."""

    anchor: BBox
    hits: int
    last_seen: int


@dataclass
class ArrivalDebouncer:
    """Count-rise hysteresis per label group, with a memory of settled spots.
    See the module docstring.

    `groups` maps labels onto one counting key (car, truck and bus as
    "vehicle"): COCO flips a parked pickup between car and truck, and counted
    per label that flip is a truck arriving. The event still carries the
    detection's own label.

    With `memory_frames` > 0, a would-be newcomer is absorbed instead of fired
    when its box sits on a known spot: one where something of its group was
    seen for at least `settle_frames` and last seen within `memory_frames`. That
    is a parked vehicle flickering out of detection and back, which the count
    alone reads as leaving and arriving. What was there on the first frame
    counts as settled (after a restart, parked cars are known at once)."""

    n_frames: int
    fire_confidence: float = 0.0  # the newcomer must reach this to raise an event
    # A count rise whose newcomer never gets confident (a ghost at 0.35 for
    # ages) is absorbed into the baseline after this many frames so it cannot
    # block later arrivals. Long, because a real arrival approaching from far
    # away spends a while below fire_confidence and must not be absorbed.
    absorb_after: int = 40
    groups: dict[str, str] = field(default_factory=dict)
    memory_frames: int = 0
    settle_frames: int = 80
    _state: dict[str, _LabelState] = field(default_factory=dict)
    _spots: dict[str, list[_Spot]] = field(default_factory=dict)
    _frames_seen: int = 0

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Feed one analysed frame. Returns the detections to raise as events
        (at most one per label group per frame)."""
        by_key: dict[str, list[Detection]] = {}
        for det in detections:
            by_key.setdefault(self.groups.get(det.label, det.label), []).append(det)
        fired: list[Detection] = []
        first_frame = self._frames_seen == 0
        frame = self._frames_seen
        self._frames_seen += 1
        for key in sorted(set(by_key) | set(self._state)):
            dets = by_key.get(key, [])
            count = len(dets)
            state = self._state.get(key)
            if state is None:
                # On the very first frame, whatever is there is the baseline:
                # nothing fires for what was already present when we started
                # looking. A label first seen later was absent before, so its
                # baseline is 0 and this frame starts its streak.
                if first_frame:
                    self._state[key] = _LabelState(baseline=count, baseline_boxes=[d.bbox for d in dets])
                    continue
                state = self._state[key] = _LabelState(baseline=0, baseline_boxes=[])
            if count > state.baseline:
                state.above += 1
                state.below = 0
                if state.above >= self.n_frames:
                    newcomer = self._newcomer(dets, state.baseline_boxes)
                    if self._on_known_spot(key, newcomer.bbox, frame):
                        pass  # a parked vehicle back from a flicker: absorb, don't fire
                    elif newcomer.confidence >= self.fire_confidence:
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
        if self.memory_frames:
            self._remember(by_key, frame, seed=first_frame)
        return fired

    def _on_known_spot(self, key: str, box: BBox, frame: int) -> bool:
        return any(
            spot.hits >= self.settle_frames and frame - spot.last_seen <= self.memory_frames
            and iou(spot.anchor, box) >= 0.5
            for spot in self._spots.get(key, ())
        )

    def _remember(self, by_key: dict[str, list[Detection]], frame: int, seed: bool) -> None:
        """After this frame's decisions: refresh the spots its boxes sit on,
        open new ones, forget spots unseen for longer than the memory."""
        for key, dets in by_key.items():
            spots = self._spots.setdefault(key, [])
            for det in dets:
                spot = max(spots, key=lambda s: iou(s.anchor, det.bbox), default=None)
                if spot is not None and iou(spot.anchor, det.bbox) >= 0.5:
                    spot.hits += 1
                    spot.last_seen = frame
                else:
                    spots.append(_Spot(det.bbox, self.settle_frames if seed else 1, frame))
        for key, spots in self._spots.items():
            self._spots[key] = [s for s in spots if frame - s.last_seen <= self.memory_frames]

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
        self.debouncer = ArrivalDebouncer(
            spec.debounce_frames, fire_confidence=spec.min_confidence, groups=spec.group_map(),
            memory_frames=round(spec.memory_s * spec.analysis_fps),
            settle_frames=max(1, round(spec.settle_s * spec.analysis_fps)),
        )
        self.frames_read = 0
        self.frames_analysed = 0

    def stream(self) -> Iterator[Frame | DetectionEvent]:
        offset = 0.0
        while True:
            if self.uri.startswith("rtsp://"):
                # RTSP over TCP unless the caller chose otherwise: UDP loses
                # packets over anything but a clean LAN, and lost packets in an
                # H.264/H.265 stream are corrupt frames for the detector.
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
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
