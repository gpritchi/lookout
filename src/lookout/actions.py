"""Action sinks.

MockSink (default): logs exactly what it would have done, structured, and keeps
the list — this is what the demo and the tests run against.

HAWebhookSink (optional, later PR): POSTs to the configured Home Assistant
webhook URL. Only active when actions.ha_webhook_url is set in config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

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
    def fire(self, action: ActionSpec, context: ActionContext) -> None: ...


class MockSink:
    def __init__(self) -> None:
        self.fired: list[tuple[ActionSpec, ActionContext]] = []

    def fire(self, action: ActionSpec, context: ActionContext) -> None:
        self.fired.append((action, context))
        extras = " ".join(f"{k}={v!r}" for k, v in action.model_dump(exclude={"type"}).items())
        via = f"step={context.step_id} answer={context.answer}" if context.step_id else "on_trigger"
        log.info(
            "ACTION %s %s  <- chain=%s camera=%s trigger=%s@%.2f %s",
            action.type, extras, context.chain_id, context.camera,
            context.trigger_label, context.trigger_ts, via,
        )
