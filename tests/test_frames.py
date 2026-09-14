"""Frame ring buffer and window sampling."""

from lookout.config import WindowSpec
from lookout.events import Frame
from lookout.frames import FrameBuffer


def fill(buffer: FrameBuffer, start: float, end: float, fps: float = 2.0) -> None:
    n = 0
    while (ts := start + n / fps) <= end + 1e-9:
        buffer.put(Frame(camera="cam", ts=round(ts, 6), ref=f"f{ts:.1f}"))
        n += 1


def test_evicts_older_than_keep():
    b = FrameBuffer(keep_s=5.0, margin_s=0.0)
    fill(b, 0.0, 20.0)
    assert len(b) == 11  # 15.0 .. 20.0 at 2 fps
    assert b.at(0.0).ts == 15.0  # nearest surviving frame


def test_window_samples_evenly_across_span():
    b = FrameBuffer(keep_s=60.0)
    fill(b, 0.0, 30.0)
    frames = b.window(center_ts=10.0, spec=WindowSpec(before_s=2, after_s=8, frames=6))
    assert [f.ts for f in frames] == [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]


def test_window_picks_nearest_when_targets_fall_between_frames():
    b = FrameBuffer(keep_s=60.0)
    fill(b, 0.0, 30.0, fps=1.0)
    frames = b.window(center_ts=10.0, spec=WindowSpec(before_s=1, after_s=1, frames=3))
    assert [f.ts for f in frames] == [9.0, 10.0, 11.0]
    frames = b.window(center_ts=10.0, spec=WindowSpec(before_s=0, after_s=1, frames=3))
    # targets 10.0, 10.5, 11.0 at 1 fps: the middle one collapses onto a neighbour, no repeats
    assert [f.ts for f in frames] == [10.0, 11.0]


def test_window_single_frame_is_nearest_to_center():
    b = FrameBuffer(keep_s=60.0)
    fill(b, 0.0, 30.0)
    frames = b.window(center_ts=10.2, spec=WindowSpec(before_s=5, after_s=5, frames=1))
    assert [f.ts for f in frames] == [10.0]


def test_window_empty_when_span_has_no_frames():
    b = FrameBuffer(keep_s=60.0)
    fill(b, 0.0, 5.0)
    assert b.window(center_ts=50.0, spec=WindowSpec(before_s=1, after_s=1, frames=4)) == []
    assert FrameBuffer(keep_s=1.0).at(0.0) is None


def test_window_truncates_to_available_footage():
    b = FrameBuffer(keep_s=60.0)
    fill(b, 0.0, 12.0)  # nothing after 12 yet
    frames = b.window(center_ts=10.0, spec=WindowSpec(before_s=2, after_s=8, frames=6))
    assert frames[0].ts == 8.0 and frames[-1].ts == 12.0
    assert len(frames) < 6
