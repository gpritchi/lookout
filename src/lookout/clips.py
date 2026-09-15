"""Clip providers: where `video_clip` payloads come from.

lookout keeps JPEG frames for image payloads, but a clip with its audio track
intact is a job for whatever already records the stream. MediaMTX (the fake
camera in fixtures/sample/mediamtx.yml, and a perfectly good real one) records
each path to disk and answers "path X from time T for D seconds" with an MP4
over its playback API. That is the whole provider: a GET.

Engine timestamps are seconds since the run started; the provider is handed
the wall-clock epoch of that zero so it can ask for absolute times.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Protocol

import httpx

log = logging.getLogger(__name__)


class ClipError(RuntimeError):
    pass


class ClipProvider(Protocol):
    def clip(self, camera: str, start_ts: float, end_ts: float) -> bytes:
        """MP4 bytes covering [start_ts, end_ts] of the camera's stream, in
        engine time. Raises ClipError if the footage cannot be produced."""
        ...


class MediaMtxClips:
    def __init__(
        self,
        paths: dict[str, tuple[str, str]],
        epoch: float,
        attempts: int = 4,
        retry_s: float = 1.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """paths: camera -> (playback base URL, MediaMTX path name).
        epoch: wall-clock time (time.time()) at engine time 0.
        The recorder flushes segments a little behind live, so a request for
        footage that ends "now" can 404 for a second; hence the retries."""
        self.paths = paths
        self.epoch = epoch
        self.attempts = attempts
        self.retry_s = retry_s
        self._http = httpx.Client(transport=transport) if transport else httpx.Client()

    def close(self) -> None:
        self._http.close()

    def clip(self, camera: str, start_ts: float, end_ts: float) -> bytes:
        try:
            base, path = self.paths[camera]
        except KeyError as exc:
            raise ClipError(f"camera '{camera}' has no clip source configured") from exc
        if end_ts <= start_ts:
            raise ClipError(f"empty clip range {start_ts}..{end_ts}")
        start = datetime.fromtimestamp(self.epoch + start_ts, tz=timezone.utc)
        params = {
            "path": path,
            "start": start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "duration": f"{end_ts - start_ts:.3f}",
            "format": "mp4",
        }
        url = base.rstrip("/") + "/get"
        last = ""
        for attempt in range(1, self.attempts + 1):
            try:
                response = self._http.get(url, params=params, timeout=30)
            except httpx.HTTPError as exc:
                last = str(exc)
            else:
                if response.status_code == 200 and response.content:
                    log.info("clip %s %.1fs -> %d KB", camera, end_ts - start_ts, len(response.content) // 1024)
                    return response.content
                last = f"HTTP {response.status_code}: {response.text[:120]}"
            if attempt < self.attempts:
                time.sleep(self.retry_s)
        raise ClipError(f"{camera}: playback API gave no clip after {self.attempts} attempts ({last})")
