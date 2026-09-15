"""Action sinks.

MockSink (default): logs exactly what it would have done, structured, and keeps
the list — this is what the demo and the tests run against.

HAWebhookSink: POSTs the action and its context as JSON to a Home Assistant
webhook trigger. HA side, an automation with a webhook trigger reads
`trigger.json` and does the notifying, TV switching, or whatever the action
type means in that house. Only active when actions.sink is "ha_webhook".

Both sinks keep `fired`, so the runner's --stop-after-actions works with either.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import httpx

from lookout.config import ActionSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionContext:
    """Why an action fired. Enough for a notification to be specific and for a
    log line to be traceable back to the chain that produced it."""

    chain_id: str
    camera: str
    trigger_label: str
    trigger_ts: float
    step_id: str | None = None
    answer: str | None = None


class ActionSink(Protocol):
    fired: list[tuple[ActionSpec, ActionContext]]

    def fire(self, action: ActionSpec, context: ActionContext) -> None: ...


def describe(action: ActionSpec, context: ActionContext) -> str:
    extras = " ".join(f"{k}={v!r}" for k, v in action.model_dump(exclude={"type"}).items())
    via = f"step={context.step_id} answer={context.answer}" if context.step_id else "on_trigger"
    return (
        f"ACTION {action.type} {extras}  <- chain={context.chain_id} camera={context.camera} "
        f"trigger={context.trigger_label}@{context.trigger_ts:.2f} {via}"
    )


class MockSink:
    def __init__(self) -> None:
        self.fired: list[tuple[ActionSpec, ActionContext]] = []

    def fire(self, action: ActionSpec, context: ActionContext) -> None:
        self.fired.append((action, context))
        log.info("%s", describe(action, context))


class HAWebhookSink:
    """One POST per action. The body is the action's own fields plus the
    context, flat, so an HA automation can template `trigger.json.message`
    or branch on `trigger.json.type` without unpacking anything.

    A failed POST is logged and counted, never raised: the chain has already
    concluded, and a dead HA must not take the engine down with it."""

    def __init__(self, url: str, timeout_s: float = 5.0, transport: httpx.BaseTransport | None = None) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.fired: list[tuple[ActionSpec, ActionContext]] = []
        self.failed = 0
        self._http = httpx.Client(transport=transport) if transport else httpx.Client()

    def close(self) -> None:
        self._http.close()

    def payload(self, action: ActionSpec, context: ActionContext) -> dict[str, Any]:
        body: dict[str, Any] = {"type": action.type, **action.model_dump(exclude={"type"})}
        body.update({f"context_{k}" if k in body else k: v for k, v in asdict(context).items()})
        return body

    def fire(self, action: ActionSpec, context: ActionContext) -> None:
        self.fired.append((action, context))
        log.info("%s", describe(action, context))
        try:
            response = self._http.post(self.url, json=self.payload(action, context), timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            self.failed += 1
            log.error("webhook %s failed: %s", self.url, exc)
            return
        if response.status_code >= 300:
            self.failed += 1
            log.error("webhook %s returned HTTP %s: %s", self.url, response.status_code, response.text[:200])
