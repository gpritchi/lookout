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


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str  # file path or RTSP URI; whatever OpenCV VideoCapture accepts
    tier1: str = "yolo"  # which tier-1 source type feeds this camera


class Tier1Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "yolov8n"
    backend: Literal["openvino", "torch"] = "openvino"
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    debounce_frames: int = Field(default=3, ge=1)


class ActionsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sink: Literal["mock", "ha_webhook"] = "mock"
    ha_webhook_url: str | None = None

    @model_validator(mode="after")
    def _webhook_needs_url(self) -> "ActionsSpec":
        if self.sink == "ha_webhook" and not self.ha_webhook_url:
            raise ValueError("actions.sink is 'ha_webhook' but actions.ha_webhook_url is not set")
        return self


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

    label: str
    priority: int = 0
    held_frames: int | None = Field(default=None, ge=1)


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


class StepSpec(BaseModel):
    """One escalation step: ask `model` about a `payload` built from the
    triggering event, wait up to `timeout_s`, map the answer through `outcomes`.
    `priority` is the queue priority of the inference job (higher wins)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    model: str
    payload: Capability
    priority: int = 0
    timeout_s: float = Field(gt=0)
    prompt: str
    outcomes: dict[str, OutcomeSpec] = Field(min_length=1)


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
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelSpec] = Field(min_length=1)
    cameras: dict[str, CameraSpec] = Field(min_length=1)
    tier1: Tier1Spec = Field(default_factory=Tier1Spec)
    actions: ActionsSpec = Field(default_factory=ActionsSpec)
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


def load_config(path: str | Path) -> Config:
    """Read a JSON config file and validate it. Raises ConfigError on anything
    wrong with the contents, and the usual OSError if the file is unreadable."""
    with Path(path).open() as fh:
        data = json.load(fh)
    return Config.from_dict(data)
