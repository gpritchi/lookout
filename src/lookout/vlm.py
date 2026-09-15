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
  video_clip      not sent by this client (a model that takes clips gets its
                  own client later); raising here lets the chain reset cleanly.

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
        if payload.kind == "video_clip":
            raise VlmClientError(f"{payload.chain_id}/{payload.step_id}: video_clip payloads need a clip-capable client")
        if not payload.frames:
            raise VlmClientError(f"{payload.chain_id}/{payload.step_id}: no frames in payload")
        for frame in payload.frames:
            if not isinstance(frame.data, (bytes, bytearray)):
                raise VlmClientError(
                    f"{payload.chain_id}/{payload.step_id}: frame {frame.ref!r} carries no image data "
                    "(replay fixtures cannot be sent to a real model)"
                )

        content: list[dict[str, Any]] = []
        if payload.kind == "image":
            content.append(_image_part(payload.frames[-1].data))
        else:
            first_ts = payload.frames[0].ts
            for i, frame in enumerate(payload.frames, 1):
                content.append({"type": "text", "text": f"Frame {i} (+{frame.ts - first_ts:.1f}s):"})
                content.append(_image_part(frame.data))
        content.append({"type": "text", "text": payload.prompt})

        return {
            "model": spec.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


def _image_part(jpeg: bytes) -> dict[str, Any]:
    b64 = base64.b64encode(jpeg).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
