"""Config loading and validation (pydantic).

One JSON file describes everything the engine needs: a models registry, cameras,
tier-1 settings, the action sink, and the chains. Loading it is the moment swap-out
mistakes surface — a step whose payload type its model cannot accept, a `next`
pointing at a step that does not exist — so all of those are checked here, at
startup, with an error naming the chain, step, and value involved.

Two layers of validation:

1. Shape, by pydantic: every field has the right type, every enum is one of its
   allowed values, every outcome does exactly one thing, a steps-less chain
   carries an `on_trigger`.
2. References, by `Config.check_references()`: rules that need the whole
   document — model exists, capability gate, `next` targets, camera exists.
   These raise `ConfigError` with a message a human can act on.

`load_config(path)` runs both. Tests build config dicts inline and call
`Config.from_dict`, which does the same thing without touching disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# The input types a model can accept, and the payload types a step can ask for.
# Same vocabulary on purpose: the capability gate is a set membership test.
Capability = Literal["image", "image_sequence", "video_clip"]


class ConfigError(ValueError):
    """A config that parsed but is not usable. The message says what to fix."""


class ModelSpec(BaseModel):
    """One entry in the models registry: where to send requests and what the
    model can consume. `endpoint` is any OpenAI-compatible base URL; `model` is
    the name the endpoint knows it by."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["openai-compatible"] = "openai-compatible"
    endpoint: str
    model: str
    capabilities: list[Capability] = Field(min_length=1)
    # Name of an environment variable holding a bearer token, if the endpoint
    # wants one. The key itself never lives in config.
    api_key_env: str | None = None


class ClipsSpec(BaseModel):
    """Where to fetch recorded footage of a camera for `video_clip` payloads:
    a MediaMTX playback endpoint and the path name it records under."""

    model_config = ConfigDict(extra="forbid")

    playback: str  # e.g. http://recorder:9996
    path: str  # the MediaMTX path, e.g. "driveway"


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str  # file path or RTSP URI; whatever OpenCV VideoCapture accepts
    tier1: str = "yolo"  # which tier-1 source type feeds this camera
    clips: ClipsSpec | None = None  # required by any chain step with a video_clip payload


class Tier1Spec(BaseModel):
    """The cheap detector and how often it looks.

    `analysis_fps` is how many frames per second the detector sees; the rest are
    skipped. `frame_fps` is how many per second go into the ring buffer for
    escalation windows, stored as JPEG no wider than `frame_max_width`.
    `debounce_frames` is the hysteresis: a label's count must sit above its
    baseline for that many analysed frames to fire, and below it for that many
    to lower the baseline again.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = "yolov8n"
    backend: Literal["openvino", "torch"] = "openvino"
    device: str | None = None  # OpenVINO device (CPU, GPU) or torch device (cpu, mps); None = default
    imgsz: int = Field(default=640, ge=64)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Objects are COUNTED down to this confidence, so a parked car hovering
    # around min_confidence does not flicker in and out and look like an
    # arrival. Only a newcomer at or above min_confidence FIRES.
    count_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    # Ignore boxes smaller than this fraction of the frame area: street traffic
    # at the far edge of a driveway camera is tiny and never worth a chain.
    min_box_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    debounce_frames: int = Field(default=3, ge=1)
    analysis_fps: float = Field(default=4.0, gt=0)
    frame_fps: float = Field(default=2.0, gt=0)
    frame_max_width: int = Field(default=1280, ge=64)
    jpeg_quality: int = Field(default=85, ge=1, le=100)

    @model_validator(mode="after")
    def _count_not_above_fire(self) -> "Tier1Spec":
        if self.count_confidence > self.min_confidence:
            raise ValueError("tier1.count_confidence must not exceed tier1.min_confidence")
        return self


class ActionsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sink: Literal["mock", "ha_webhook"] = "mock"
    ha_webhook_url: str | None = None  # https://<ha>/api/webhook/<id>
    ha_timeout_s: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _webhook_needs_url(self) -> "ActionsSpec":
        if self.sink == "ha_webhook" and not self.ha_webhook_url:
            raise ValueError("actions.sink is 'ha_webhook' but actions.ha_webhook_url is not set")
        return self


class MetricsSpec(BaseModel):
    """Serve prometheus_client's registry over HTTP. Off unless a port is set;
    the histogram is recorded either way."""

    model_config = ConfigDict(extra="forbid")

    port: int | None = Field(default=None, ge=1, le=65535)
    host: str = "0.0.0.0"


class ActionSpec(BaseModel):
    """What to fire when a chain concludes. `type` selects the handler; the sink
    decides what the other fields mean (a notify wants `message`, a webhook wants
    `service`), so anything beyond `type` is passed through untouched."""

    model_config = ConfigDict(extra="allow")

    type: str


class TriggerSpec(BaseModel):
    """The tier-1 condition that starts a chain. `label` is a detector label
    (COCO class or a `gesture:*` name); `held_frames` is the optional intent
    filter for one-shot chains; `priority` orders competing chain starts."""

    model_config = ConfigDict(extra="forbid")

    label: str | list[str]  # one label, or any of several (COCO flips a pickup between car and truck)
    priority: int = 0
    held_frames: int | None = Field(default=None, ge=1)

    @property
    def labels(self) -> list[str]:
        return [self.label] if isinstance(self.label, str) else self.label

    def matches(self, label: str) -> bool:
        return label in self.labels


class OutcomeSpec(BaseModel):
    """What a step does with one of its answers. Exactly one of: go to `next`,
    `end` the chain, fire an `action`, or `retry_in_s` seconds later."""

    model_config = ConfigDict(extra="forbid")

    next: str | None = None
    end: bool = False
    action: ActionSpec | None = None
    retry_in_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _exactly_one(self) -> "OutcomeSpec":
        chosen = [
            name
            for name, value in (
                ("next", self.next),
                ("end", self.end or None),
                ("action", self.action),
                ("retry_in_s", self.retry_in_s),
            )
            if value is not None
        ]
        if len(chosen) != 1:
            raise ValueError(
                "an outcome must set exactly one of next / end / action / retry_in_s, "
                f"got {chosen or 'none'}"
            )
        return self


class WindowSpec(BaseModel):
    """How much of the stream around the trigger a step looks at.

    Relative to the triggering event's timestamp: `before_s` seconds back,
    `after_s` seconds forward, sampled evenly into `frames` frames. An `image`
    payload ignores this and uses the trigger frame; `image_sequence` and
    `video_clip` require it. `after_s` implies waiting: the engine cannot
    dispatch the step until that much stream has been seen.
    """

    model_config = ConfigDict(extra="forbid")

    before_s: float = Field(default=0.0, ge=0.0)
    after_s: float = Field(default=0.0, ge=0.0)
    frames: int = Field(default=1, ge=1)

    @property
    def span_s(self) -> float:
        return self.before_s + self.after_s


class StepSpec(BaseModel):
    """One escalation step: ask `model` about a `payload` built from the
    triggering event, wait up to `timeout_s`, map the answer through `outcomes`.
    `priority` is the queue priority of the inference job (higher wins).
    `window` says how much stream the payload covers; a chain-level `window`
    is the default for steps that leave it out."""

    model_config = ConfigDict(extra="forbid")

    id: str
    model: str
    payload: Capability
    priority: int = 0
    timeout_s: float = Field(gt=0)
    prompt: str
    outcomes: dict[str, OutcomeSpec] = Field(min_length=1)
    window: WindowSpec | None = None


class OnTriggerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ActionSpec


class ChainSpec(BaseModel):
    """A chain is a trigger plus an ordered list of steps. The first step is the
    entry point; later steps are reached only via `next`. A chain with no steps
    fires `on_trigger` directly — the one-shot case where tier-1 is enough."""

    model_config = ConfigDict(extra="forbid")

    id: str
    camera: str
    trigger: TriggerSpec
    steps: list[StepSpec] = Field(default_factory=list)
    on_trigger: OnTriggerSpec | None = None
    window: WindowSpec | None = None
    # How long a run may live from its trigger before the engine gives up on
    # it. Bounds `retry_in_s` loops: a step that keeps answering NOBODY_YET
    # would otherwise hold the chain forever and block the next real trigger
    # (seen live 2026-09-14: a seam-triggered run sat on retries while the
    # actual arrival went by as "already active, ignoring").
    # Size it for the chain's slowest path: a clip step can spend a minute
    # waiting for its window and three more in inference, and an answer that
    # lands after the run was abandoned is dropped as stale (seen live).
    max_age_s: float = Field(default=600.0, gt=0)
    comment: str | None = None

    @model_validator(mode="after")
    def _steps_or_on_trigger(self) -> "ChainSpec":
        if not self.steps and self.on_trigger is None:
            raise ValueError(f"chain '{self.id}' has no steps and no on_trigger; it would never do anything")
        if self.steps and self.on_trigger is not None:
            raise ValueError(f"chain '{self.id}' has both steps and on_trigger; pick one")
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"chain '{self.id}' has duplicate step id '{step.id}'")
            seen.add(step.id)
            if step.payload != "image" and self.window_for(step) is None:
                raise ValueError(
                    f"chain '{self.id}' step '{step.id}' has payload '{step.payload}' but no window; "
                    "set `window` on the step or the chain"
                )
        return self

    def window_for(self, step: StepSpec) -> WindowSpec | None:
        """The step's own window, else the chain default, else None."""
        return step.window if step.window is not None else self.window

    def step(self, step_id: str) -> StepSpec:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)

    @property
    def buffer_seconds(self) -> float:
        """How much stream this chain can ever look back over: the widest
        window's before_s. Frames after the trigger accumulate as they arrive,
        so after_s does not need buffering ahead of time."""
        return max((w.before_s for w in (self.window_for(s) for s in self.steps) if w is not None), default=0.0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelSpec] = Field(min_length=1)
    cameras: dict[str, CameraSpec] = Field(min_length=1)
    tier1: Tier1Spec = Field(default_factory=Tier1Spec)
    actions: ActionsSpec = Field(default_factory=ActionsSpec)
    metrics: MetricsSpec = Field(default_factory=MetricsSpec)
    chains: list[ChainSpec] = Field(min_length=1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Parse and fully validate. Shape errors come back as ConfigError too,
        so callers handle one exception type."""
        try:
            config = cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc
        config.check_references()
        return config

    def check_references(self) -> None:
        """The cross-document rules. Each failure names enough to fix it."""
        seen_chain_ids: set[str] = set()
        for chain in self.chains:
            if chain.id in seen_chain_ids:
                raise ConfigError(f"duplicate chain id '{chain.id}'")
            seen_chain_ids.add(chain.id)

            if chain.camera not in self.cameras:
                raise ConfigError(
                    f"chain '{chain.id}' is bound to camera '{chain.camera}', "
                    f"which is not in cameras {sorted(self.cameras)}"
                )
            camera = self.cameras[chain.camera]
            step_ids = {step.id for step in chain.steps}
            for step in chain.steps:
                model = self.models.get(step.model)
                if model is None:
                    raise ConfigError(
                        f"chain '{chain.id}' step '{step.id}' references model '{step.model}', "
                        f"which is not in models {sorted(self.models)}"
                    )
                if step.payload not in model.capabilities:
                    raise ConfigError(
                        f"chain '{chain.id}' step '{step.id}' sends payload '{step.payload}' "
                        f"to model '{step.model}', which only accepts {model.capabilities}"
                    )
                if step.payload == "video_clip" and camera.clips is None:
                    raise ConfigError(
                        f"chain '{chain.id}' step '{step.id}' needs a video_clip of camera "
                        f"'{chain.camera}', but that camera has no `clips` source configured"
                    )
                for answer, outcome in step.outcomes.items():
                    if outcome.next is not None and outcome.next not in step_ids:
                        raise ConfigError(
                            f"chain '{chain.id}' step '{step.id}' outcome '{answer}' goes to "
                            f"'{outcome.next}', which is not a step in that chain {sorted(step_ids)}"
                        )

    def chain(self, chain_id: str) -> ChainSpec:
        for chain in self.chains:
            if chain.id == chain_id:
                return chain
        raise KeyError(chain_id)

    def buffer_seconds(self, camera: str) -> float:
        """Ring-buffer depth a camera needs: the deepest look-back any of its
        chains asks for. Sized from config, nothing else to tune."""
        return max((c.buffer_seconds for c in self.chains if c.camera == camera), default=0.0)


def load_config(path: str | Path) -> Config:
    """Read a JSON config file and validate it. Raises ConfigError on anything
    wrong with the contents, and the usual OSError if the file is unreadable."""
    with Path(path).open() as fh:
        data = json.load(fh)
    return Config.from_dict(data)
