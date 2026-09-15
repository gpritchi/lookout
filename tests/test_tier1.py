"""Debounce basics, and VideoSource over a tiny synthetic clip with a fake
detector (no model weights needed)."""

import cv2
import numpy as np
import pytest

from lookout.config import Tier1Spec
from lookout.events import DetectionEvent, Frame
from lookout.tier1 import ArrivalDebouncer, Detection, VideoSource, iou

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
