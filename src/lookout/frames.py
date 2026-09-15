"""Per-camera frame ring buffer and window sampling.

Escalation payloads are not the frame the detector fired on; they are a slice
of the stream around it, sized by the step's WindowSpec. The buffer keeps the
last `keep_s` seconds of frames per camera and answers "give me N frames evenly
spread over [center - before_s, center + after_s]".

`FrameBuffer.window()` is where "how much of the surrounding stream is watched"
is decided; WindowSpec in the config is where it is set.
"""

from __future__ import annotations

import bisect
import threading
from collections import deque

from lookout.config import WindowSpec
from lookout.events import Frame


class FrameBuffer:
    def __init__(self, keep_s: float, margin_s: float = 2.0) -> None:
        """Keep frames newer than (latest.ts - keep_s - margin_s). The margin
        covers timestamp jitter between the source and the engine."""
        self.keep_s = keep_s + margin_s
        self._frames: deque[Frame] = deque()
        self._lock = threading.Lock()

    def put(self, frame: Frame) -> None:
        with self._lock:
            self._frames.append(frame)
            horizon = frame.ts - self.keep_s
            while self._frames and self._frames[0].ts < horizon:
                self._frames.popleft()

    def __len__(self) -> int:
        return len(self._frames)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def at(self, ts: float) -> Frame | None:
        """The frame nearest to ts, or None if the buffer is empty."""
        with self._lock:
            return self._nearest_locked(list(self._frames), ts)

    def window(self, center_ts: float, spec: WindowSpec) -> list[Frame]:
        """`spec.frames` frames spread evenly over [center - before_s,
        center + after_s], each the nearest available frame to its target
        instant, without repeats. Fewer come back if the buffer is sparse."""
        start, end = center_ts - spec.before_s, center_ts + spec.after_s
        with self._lock:
            candidates = [f for f in self._frames if start <= f.ts <= end]
        if not candidates:
            return []
        if spec.frames == 1:
            return [self._nearest_locked(candidates, center_ts)]  # type: ignore[list-item]
        targets = [start + (end - start) * i / (spec.frames - 1) for i in range(spec.frames)]
        chosen: list[Frame] = []
        for target in targets:
            frame = self._nearest_locked(candidates, target)
            if frame is not None and (not chosen or frame.ts > chosen[-1].ts):
                chosen.append(frame)
        return chosen

    @staticmethod
    def _nearest_locked(frames: list[Frame], ts: float) -> Frame | None:
        if not frames:
            return None
        stamps = [f.ts for f in frames]
        i = bisect.bisect_left(stamps, ts)
        if i == 0:
            return frames[0]
        if i == len(frames):
            return frames[-1]
        before, after = frames[i - 1], frames[i]
        return before if ts - before.ts <= after.ts - ts else after
