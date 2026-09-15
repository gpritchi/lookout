"""Tier-2 client: turn a StepPayload into an OpenAI-compatible chat request.

One client serves every model in the registry; the job's model name selects
the endpoint and the name the endpoint knows the model by. Anything that
speaks the OpenAI chat API works — in the demo it is a LiteLLM proxy in front
of llama.cpp, but nothing here knows or cares.

Payload shapes:

  image           one image part, then the prompt.
  image_sequence  "Frame N (+T.Ts):" text, image, ... then the prompt. The text
                  between images is not decoration: llama.cpp merges strictly
                  consecutive image parts into one "video" input, so without a
                  separator a six-frame window can arrive as three frames.
  video_clip      one `input_video` part carrying the MP4 as a data URL, then
                  the prompt. This is llama.cpp's own content type (and the
                  llama.cpp-omni fork's, which also reads the clip's audio
                  track); the OpenAI API has no video part, so a proxy in the
                  middle has to pass it through untouched.

Frames must carry JPEG bytes (`Frame.data`). Replay fixtures have none, and
that is a configuration error at this layer, not something to paper over.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import httpx

from lookout.config import ModelSpec
from lookout.engine import StepPayload
from lookout.scheduler import InferenceJob

log = logging.getLogger(__name__)


class VlmClientError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(
        self,
        models: dict[str, ModelSpec],
        max_tokens: int = 64,
        temperature: float = 0.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.models = models
        self.max_tokens = max_tokens
        self.temperature = temperature
        # One pooled client; per-call timeout comes from the step.
        self._http = httpx.Client(transport=transport) if transport else httpx.Client()

    def close(self) -> None:
        self._http.close()

    # -- the Worker's `run` callable -----------------------------------------

    def __call__(self, job: InferenceJob) -> str:
        spec = self.models.get(job.model)
        if spec is None:
            raise VlmClientError(f"job for unknown model '{job.model}'")
        payload: StepPayload = job.payload
        body = self.build_request(spec, payload)
        headers = {}
        if spec.api_key_env and os.environ.get(spec.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[spec.api_key_env]}"
        url = spec.endpoint.rstrip("/") + "/chat/completions"
        try:
            response = self._http.post(url, json=body, headers=headers, timeout=payload.timeout_s)
        except httpx.HTTPError as exc:
            raise VlmClientError(f"{job.model}: {exc}") from exc
        if response.status_code != 200:
            raise VlmClientError(f"{job.model}: HTTP {response.status_code}: {response.text[:200]}")
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise VlmClientError(f"{job.model}: malformed response: {response.text[:200]}") from exc
        usage = data.get("usage") or {}
        log.info(
            "%s/%s -> %s: %r (%s prompt tokens)",
            payload.chain_id, payload.step_id, job.model, text.strip(), usage.get("prompt_tokens", "?"),
        )
        return text

    # -- request building, kept separate so tests can see it -----------------

    def build_request(self, spec: ModelSpec, payload: StepPayload) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if payload.kind == "video_clip":
            if not payload.clip:
                raise VlmClientError(f"{payload.chain_id}/{payload.step_id}: video_clip payload carries no clip")
            content.append(_video_part(payload.clip))
            content.append({"type": "text", "text": payload.prompt})
            return self._body(spec, content)

        if not payload.frames:
            raise VlmClientError(f"{payload.chain_id}/{payload.step_id}: no frames in payload")
        for frame in payload.frames:
            if not isinstance(frame.data, (bytes, bytearray)):
                raise VlmClientError(
                    f"{payload.chain_id}/{payload.step_id}: frame {frame.ref!r} carries no image data "
                    "(replay fixtures cannot be sent to a real model)"
                )

        if payload.kind == "image":
            content.append(_image_part(payload.frames[-1].data))
        else:
            first_ts = payload.frames[0].ts
            for i, frame in enumerate(payload.frames, 1):
                content.append({"type": "text", "text": f"Frame {i} (+{frame.ts - first_ts:.1f}s):"})
                content.append(_image_part(frame.data))
        content.append({"type": "text", "text": payload.prompt})
        return self._body(spec, content)

    def _body(self, spec: ModelSpec, content: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": spec.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


def _image_part(jpeg: bytes) -> dict[str, Any]:
    b64 = base64.b64encode(jpeg).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _video_part(mp4: bytes) -> dict[str, Any]:
    # Raw base64, not a data: URL. llama.cpp's input_video path decodes the
    # whole string as base64 (the data:-URL parsing belongs to image_url only),
    # so a prefixed value turns into a few garbage bytes and "failed to decode
    # buffer as either image/audio/video".
    return {"type": "input_video", "input_video": {"data": base64.b64encode(mp4).decode("ascii")}}
