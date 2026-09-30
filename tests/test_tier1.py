"""Debounce basics, VideoSource over a tiny synthetic clip with a fake
detector, and the OpenVINO detector's pre/postprocessing, down to a whole
detector call on a tiny synthetic OpenVINO model (no model weights needed)."""

import sys

import cv2
import numpy as np
import pytest

from lookout.config import Tier1Spec
from lookout.events import DetectionEvent, Frame
from lookout.tier1 import (
    ArrivalDebouncer,
    Detection,
    DetectorError,
    OpenVinoDetector,
    VideoSource,
    decode,
    iou,
    letterbox,
    load_detector,
    nms,
    to_input,
    unletterbox,
)

CAR_A = Detection("car", 0.8, (10, 10, 50, 50))
CAR_B = Detection("car", 0.7, (100, 100, 150, 150))
PERSON = Detection("person", 0.9, (60, 20, 80, 70))


def test_parked_car_never_fires():
    d = ArrivalDebouncer(n_frames=3)
    assert all(d.update([CAR_A]) == [] for _ in range(20))


def test_new_car_fires_once_after_n_frames_and_names_the_newcomer():
    d = ArrivalDebouncer(n_frames=3)
    d.update([CAR_A])
    assert d.update([CAR_A, CAR_B]) == []
    assert d.update([CAR_A, CAR_B]) == []
    assert d.update([CAR_A, CAR_B]) == [CAR_B]
    assert all(d.update([CAR_A, CAR_B]) == [] for _ in range(10))


def test_flicker_below_n_does_not_fire():
    d = ArrivalDebouncer(n_frames=3)
    d.update([CAR_A])
    d.update([CAR_A, CAR_B])
    d.update([CAR_A, CAR_B])
    assert d.update([CAR_A]) == []  # streak broken
    assert d.update([CAR_A, CAR_B]) == []
    assert d.update([CAR_A, CAR_B]) == []
    assert d.update([CAR_A, CAR_B]) == [CAR_B]


def test_leaving_lowers_baseline_so_return_fires_again():
    d = ArrivalDebouncer(n_frames=2)
    d.update([CAR_A, CAR_B])
    for _ in range(2):
        d.update([CAR_A])  # B leaves
    assert d.update([CAR_A, CAR_B]) == []
    assert d.update([CAR_A, CAR_B]) == [CAR_B]


def test_labels_are_independent():
    d = ArrivalDebouncer(n_frames=2)
    d.update([CAR_A])
    d.update([CAR_A, PERSON])
    assert d.update([CAR_A, PERSON]) == [PERSON]


def test_label_present_at_start_is_baseline_but_later_first_sight_fires():
    d = ArrivalDebouncer(n_frames=2)
    d.update([CAR_A])  # first frame: the car is scenery
    assert d.update([CAR_A]) == []
    assert d.update([CAR_A, PERSON]) == []  # person never seen before: baseline 0, streak 1
    assert d.update([CAR_A, PERSON]) == [PERSON]
    assert d.update([CAR_A, PERSON]) == []
    d.update([CAR_A])
    d.update([CAR_A])  # person gone for n frames: baseline back to 0
    d.update([CAR_A, PERSON])
    assert d.update([CAR_A, PERSON]) == [PERSON]


def test_weak_newcomer_waits_and_fires_once_confident():
    """An arrival approaching from far away is counted early at low confidence
    and must fire when it becomes convincing, not be absorbed as scenery."""
    d = ArrivalDebouncer(n_frames=2, fire_confidence=0.5)
    weak = Detection("car", 0.35, CAR_B.bbox)
    d.update([CAR_A])
    d.update([CAR_A, weak])
    assert d.update([CAR_A, weak]) == []  # rise seen, newcomer too weak: keep watching
    assert d.update([CAR_A, weak]) == []
    assert d.update([CAR_A, CAR_B]) == [CAR_B]  # same box, now confident
    assert d.update([CAR_A, CAR_B]) == []


def test_persistent_ghost_is_absorbed_and_cannot_block_a_real_arrival():
    d = ArrivalDebouncer(n_frames=2, fire_confidence=0.5, absorb_after=5)
    ghost = Detection("car", 0.35, CAR_B.bbox)
    d.update([CAR_A])
    for _ in range(5):
        assert d.update([CAR_A, ghost]) == []
    strong = Detection("car", 0.9, (300, 300, 340, 340))
    d.update([CAR_A, ghost, strong])
    assert d.update([CAR_A, ghost, strong]) == [strong]


def test_newcomer_is_most_confident_non_overlapping_box():
    d = ArrivalDebouncer(n_frames=1)
    d.update([CAR_A])
    weak_new = Detection("car", 0.4, (300, 300, 340, 340))
    strong_new = Detection("car", 0.9, (400, 400, 440, 440))
    assert d.update([CAR_A, weak_new, strong_new]) == [strong_new]


# -- spot memory and label groups: the soak's parked-vehicle re-triggers -------
#
# 2026-09-30 soak on the real camera: 399 triggers in 11 h, 94% of them from the
# same few boxes. The top one (159 triggers) was a white pickup parked on the
# street whose detection flickers and whose label flips car <-> truck.

PICKUP = Detection("car", 0.8, (89, 149, 519, 382))
PICKUP_AS_TRUCK = Detection("truck", 0.5, (89, 149, 519, 382))
SEDAN = Detection("car", 0.8, (1667, 213, 1915, 363))
VEHICLES = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle"}


def settled(n_frames=2, memory=400, settle=80, groups=None):
    return ArrivalDebouncer(n_frames=n_frames, groups=groups or {}, memory_frames=memory, settle_frames=settle)


def test_parked_car_flicker_refires_without_memory():
    """The soak's failure, pinned: detection drops for n frames, the baseline
    follows it down, and the same parked car comes back as a new arrival."""
    d = ArrivalDebouncer(n_frames=2)
    d.update([PICKUP])
    for _ in range(100):
        d.update([PICKUP])
    d.update([])
    d.update([])  # flickered out long enough to lower the baseline
    d.update([PICKUP])
    assert d.update([PICKUP]) == [PICKUP]


def test_parked_car_flicker_is_absorbed_with_memory():
    d = settled()
    for _ in range(100):
        d.update([SEDAN])  # parked at the curb from the start
    for _ in range(3):
        d.update([])
    d.update([SEDAN])
    assert d.update([SEDAN]) == []
    assert all(d.update([SEDAN]) == [] for _ in range(10))


def test_label_flip_on_one_vehicle_is_not_an_arrival_with_groups():
    d = settled(groups=VEHICLES)
    d.update([PICKUP])
    for _ in range(100):
        d.update([PICKUP])
    for _ in range(3):
        d.update([PICKUP_AS_TRUCK])  # same box, other label
    assert all(d.update([PICKUP_AS_TRUCK]) == [] for _ in range(5))
    assert all(d.update([PICKUP]) == [] for _ in range(5))


def test_label_flip_without_groups_fires_as_a_truck_arrival():
    d = ArrivalDebouncer(n_frames=2)
    d.update([PICKUP])
    d.update([PICKUP_AS_TRUCK])
    assert d.update([PICKUP_AS_TRUCK]) == [PICKUP_AS_TRUCK]


def test_first_frame_seeds_the_memory():
    """After a restart, what's parked is known at once: its first flicker must
    not fire just because it hasn't sat there for settle frames yet."""
    d = settled()
    d.update([PICKUP])  # first frame after a pod restart
    d.update([])
    d.update([])
    d.update([PICKUP])
    assert d.update([PICKUP]) == []


def test_new_vehicle_at_a_new_spot_still_fires_with_memory():
    d = settled(groups=VEHICLES)
    d.update([PICKUP])
    for _ in range(100):
        d.update([PICKUP])
    van = Detection("car", 0.88, (735, 75, 1229, 274))
    d.update([PICKUP, van])
    assert d.update([PICKUP, van]) == [van]


def test_a_moving_vehicle_does_not_make_its_stopping_place_known():
    """The Amazon van drives in, slows, stops: the spots it passed through are
    brief, so where it stops is still new when the rise is judged."""
    d = settled(n_frames=3, groups=VEHICLES)
    d.update([PICKUP])
    for _ in range(100):
        d.update([PICKUP])
    positions = [(1500, 90), (1350, 130), (1200, 160), (1080, 172), (1000, 174), (985, 175)]
    fired = []
    for cx, cy in positions:
        fired += d.update([PICKUP, Detection("car", 0.88, (cx - 247, cy - 100, cx + 247, cy + 100))])
    for _ in range(3):
        fired += d.update([PICKUP, Detection("car", 0.88, (985 - 247, 75, 985 + 247, 275))])
    assert len(fired) == 1 and fired[0].label == "car"


def test_a_known_spot_is_forgotten_after_memory_frames():
    d = settled(memory=50)
    for _ in range(100):
        d.update([SEDAN])
    for _ in range(60):
        d.update([])  # gone longer than the memory
    d.update([SEDAN])
    assert d.update([SEDAN]) == [SEDAN]


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


@pytest.fixture
def tiny_clip(tmp_path):
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
    assert writer.isOpened()
    for i in range(30):
        frame = np.full((96, 128, 3), i * 8 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_video_source_frames_and_events(tiny_clip):
    calls = []

    def fake_detector(frame):
        calls.append(frame.mean())
        n = len(calls)
        return [CAR_A] if n < 5 else [CAR_A, CAR_B]  # second car appears at the 5th analysed frame

    spec = Tier1Spec(analysis_fps=5, frame_fps=2, debounce_frames=2, frame_max_width=64)
    src = VideoSource("cam", str(tiny_clip), fake_detector, spec)
    items = list(src.stream())
    frames = [i for i in items if isinstance(i, Frame)]
    events = [i for i in items if isinstance(i, DetectionEvent)]
    assert src.frames_read == 30
    assert src.frames_analysed == 15  # 10 fps / 5
    assert len(frames) == 6  # 10 fps / 2 over 3 s
    assert frames[0].data[:2] == b"\xff\xd8"  # JPEG magic
    assert frames[1].ts == pytest.approx(0.5)
    assert [(e.label, e.bbox) for e in events] == [("car", CAR_B.bbox)]
    assert events[0].ts == pytest.approx(1.0)  # 6th analysed frame = index 10 at 10 fps
    # stored frames are downscaled: decode and check width
    decoded = cv2.imdecode(np.frombuffer(frames[0].data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 64


def test_video_source_loop_offsets_timestamps(tiny_clip):
    spec = Tier1Spec(analysis_fps=1, frame_fps=1)
    src = VideoSource("cam", str(tiny_clip), lambda f: [], spec, loop=True)
    stamps = []
    for item in src.stream():
        stamps.append(item.ts)
        if len(stamps) >= 5:
            break
    assert stamps == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])  # continues past the 3 s clip


# --- OpenVINO detector: pre/postprocessing -----------------------------------


def test_letterbox_16x9_frame_pads_top_and_bottom_evenly():
    frame = np.full((1080, 1920, 3), 7, dtype=np.uint8)
    image, gain, pad = letterbox(frame, (640, 640))
    assert image.shape == (640, 640, 3)
    assert gain == pytest.approx(1 / 3)
    assert pad == (0, 140)  # 1920x1080 -> 640x360, 280 rows of padding split evenly
    assert (image[:140] == 114).all() and (image[500:] == 114).all()
    assert (image[140:500] == 7).all()


def test_letterbox_odd_padding_puts_the_extra_pixel_bottom_right():
    frame = np.zeros((61, 100, 3), dtype=np.uint8)
    image, gain, pad = letterbox(frame, (64, 64))
    assert gain == pytest.approx(0.64)
    # 100x61 -> 64x39: 25 rows of padding, 12 on top and 13 below.
    assert pad == (0, 12)
    assert image.shape == (64, 64, 3)
    assert (image[:12] == 114).all() and (image[12:51] == 0).all() and (image[51:] == 114).all()


def test_letterbox_portrait_pads_left_and_right():
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    image, gain, pad = letterbox(frame, (320, 320))
    assert (gain, pad, image.shape) == (0.5, (40, 0), (320, 320, 3))


def test_to_input_is_rgb_nchw_unit_float():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 10, 20, 255  # B, G, R
    tensor = to_input(image)
    assert tensor.shape == (1, 3, 2, 3) and tensor.dtype == np.float32
    assert tensor[0, 0, 0, 0] == 1.0  # R first
    assert tensor[0, 1, 0, 0] == pytest.approx(20 / 255)
    assert tensor[0, 2, 0, 0] == pytest.approx(10 / 255)


def test_unletterbox_inverts_the_letterbox_and_clips_to_the_frame():
    frame_shape = (1080, 1920, 3)
    gain, pad = 1 / 3, (0, 140)
    in_frame = np.array([[300.0, 150.0, 900.0, 600.0]], dtype=np.float32)
    in_model = in_frame * gain + np.array([pad[0], pad[1], pad[0], pad[1]], dtype=np.float32)
    assert unletterbox(in_model, gain, pad, frame_shape) == pytest.approx(in_frame, abs=1e-3)
    # A box running into the padding is clipped to the picture.
    spill = np.array([[-5.0, 100.0, 700.0, 520.0]], dtype=np.float32)
    assert unletterbox(spill, gain, pad, frame_shape).tolist() == [[0.0, 0.0, 1920.0, 1080.0]]


def test_nms_drops_overlaps_above_threshold_keeps_at_or_below():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [0, 0, 10, 5], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    # box1 vs box0: IoU 9/11 > 0.7, dropped. box2 vs box0: IoU exactly 0.5, kept at 0.5.
    assert nms(boxes, scores, 0.5).tolist() == [0, 2, 3]
    assert nms(boxes, scores, 0.49).tolist() == [0, 3]


def test_nms_handles_zero_area_boxes_without_warnings():
    boxes = np.array([[5, 5, 5, 5], [5, 5, 5, 5]], dtype=np.float32)
    assert nms(boxes, np.array([0.9, 0.8], dtype=np.float32), 0.7).tolist() == [0, 1]


def _head(*anchors: tuple[float, float, float, float, int, float], classes: int = 3) -> np.ndarray:
    """A YOLOv8 head output [4 + classes, anchors] from (cx, cy, w, h, class, score)."""
    pred = np.zeros((4 + classes, len(anchors)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(anchors):
        pred[:4, i] = cx, cy, w, h
        pred[4 + cls, i] = score
    return pred


def test_decode_thresholds_strictly_and_converts_to_xyxy():
    pred = _head((50, 50, 20, 10, 2, 0.9), (200, 200, 20, 20, 0, 0.3), (300, 300, 20, 20, 1, 0.1))
    boxes, confs, classes = decode(pred, conf_thres=0.3)
    assert boxes.tolist() == [[40, 45, 60, 55]]  # the 0.3 exactly at the threshold does not pass
    assert confs.tolist() == pytest.approx([0.9]) and classes.tolist() == [2]


def test_decode_nms_is_per_class_and_score_ordered():
    pred = _head(
        (50, 50, 20, 20, 2, 0.6),  # car
        (51, 50, 20, 20, 2, 0.8),  # same car, better score: this one survives
        (50, 50, 20, 20, 0, 0.7),  # a person in the same place: a different class, kept
    )
    boxes, confs, classes = decode(pred, conf_thres=0.3)
    assert classes.tolist() == [2, 0]
    assert confs.tolist() == pytest.approx([0.8, 0.7])
    assert boxes[0].tolist() == [41, 40, 61, 60]


def test_decode_keeps_only_each_anchors_best_class():
    pred = _head((50, 50, 20, 20, 2, 0.8))
    pred[4 + 1, 0] = 0.5  # the same anchor also scores bicycle, lower
    _, confs, classes = decode(pred, conf_thres=0.3)
    assert classes.tolist() == [2] and confs.tolist() == pytest.approx([0.8])


def test_decode_with_nothing_above_threshold_is_empty():
    boxes, confs, classes = decode(_head((50, 50, 20, 20, 2, 0.2)), conf_thres=0.3)
    assert boxes.shape == (0, 4) and len(confs) == len(classes) == 0


# --- OpenVINO detector: a whole call on a synthetic model ---------------------


@pytest.fixture
def tiny_export(tmp_path):
    """A 64x64-input OpenVINO model whose head output is fixed: a car, a
    near-duplicate of it, and a person below the count threshold. Written the
    way Ultralytics writes an export: <name>.xml/.bin plus metadata.yaml."""
    import openvino as ov
    import openvino.opset13 as ops

    head = _head(
        (32, 32, 20, 16, 2, 0.9),
        (33, 32, 20, 16, 2, 0.8),  # IoU with the first > 0.7: suppressed
        (10, 10, 4, 4, 0, 0.2),  # below count_confidence
    )[None]
    x = ops.parameter([1, 3, 64, 64], np.float32, name="images")
    zero = ops.multiply(ops.reduce_mean(x, [1, 2, 3], keep_dims=False), ops.constant(np.float32(0)))
    out = ops.add(ops.constant(head), zero)
    export = tmp_path / "tiny_openvino_model"
    export.mkdir()
    ov.save_model(ov.Model([ops.result(out)], [x], "tiny"), str(export / "tiny.xml"), compress_to_fp16=False)
    (export / "metadata.yaml").write_text("imgsz: [64, 64]\nnames:\n  0: person\n  1: bicycle\n  2: car\n")
    return export


def test_openvino_detector_end_to_end_on_a_synthetic_model(tiny_export):
    spec = Tier1Spec(model="tiny", imgsz=64, count_confidence=0.3)
    detector = OpenVinoDetector(spec, tiny_export)
    assert detector.name == "tiny-openvino"
    # 128x96 frame -> gain 0.5, 64x48 picture, 8 rows of padding top and bottom.
    # The car (22, 24, 42, 40) in model pixels is (44, 32, 84, 64) in the frame.
    dets = detector(np.zeros((96, 128, 3), dtype=np.uint8))
    assert [(d.label, d.bbox) for d in dets] == [("car", (44, 32, 84, 64))]
    # OpenVINO's ARM CPU plugin computes in f16 by default (0.9 -> 0.8999), x86 in f32.
    assert dets[0].confidence == pytest.approx(0.9, abs=1e-3)


def test_openvino_detector_applies_min_box_frac(tiny_export):
    spec = Tier1Spec(model="tiny", imgsz=64, count_confidence=0.3, min_box_frac=0.2)  # car is 1280 of 12288 px
    assert OpenVinoDetector(spec, tiny_export)(np.zeros((96, 128, 3), dtype=np.uint8)) == []


def test_openvino_detector_rejects_an_export_at_another_size(tiny_export):
    with pytest.raises(DetectorError, match="exported at 64x64 but tier1.imgsz is 640"):
        OpenVinoDetector(Tier1Spec(model="tiny"), tiny_export)


def test_openvino_detector_rejects_an_incomplete_export(tiny_export):
    (tiny_export / "metadata.yaml").unlink()
    with pytest.raises(DetectorError, match="not a complete OpenVINO export"):
        OpenVinoDetector(Tier1Spec(model="tiny", imgsz=64), tiny_export)


def test_load_detector_uses_an_existing_export(tiny_export):
    detector = load_detector(Tier1Spec(model="tiny", imgsz=64), models_dir=tiny_export.parent)
    assert isinstance(detector, OpenVinoDetector)


def test_missing_export_without_ultralytics_says_how_to_export(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # import fails, as in the image
    with pytest.raises(DetectorError, match=r"no OpenVINO export at .*uv sync --extra torch"):
        load_detector(Tier1Spec(), models_dir=tmp_path)


def test_torch_backend_without_ultralytics_says_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(DetectorError, match=r'backend "torch" needs .*uv sync --extra torch'):
        load_detector(Tier1Spec(backend="torch"), models_dir=tmp_path)
