"""Chain engine: the state machine between tier-1 events and inference jobs.

One ChainRun per active chain. A run is armed on a matching DetectionEvent,
walks its steps by submitting InferenceJobs to the scheduler and mapping the
answers through each step's outcomes, and ends when an outcome says `end`,
fires an `action`, times out, or comes back with an answer the config does not
recognise.

Time is injected (`clock`) so the same engine runs on a virtual clock for the
offline replay demo and on the wall clock live. Everything time-dependent goes
through tick(): dispatching steps whose window has been seen, and expiring runs
whose step exceeded its timeout_s.

The window rule, since it is the part that decides what the model sees:
  - `image` payload: the single frame nearest the run's centre timestamp.
  - `image_sequence` / `video_clip`: `window.frames` frames spread over
    [centre - before_s, centre + after_s]. The step cannot dispatch until the
    clock reaches centre + after_s, because that footage does not exist yet.
  - The centre is the trigger timestamp for the first step and for `next`
    steps. A `retry_in_s` re-centres so the fresh window ends at retry time.
"""

from __future__ import annotations

import itertools
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lookout.actions import ActionContext, ActionSink
from lookout.clips import ClipError, ClipProvider
from lookout.config import Capability, ChainSpec, Config, OutcomeSpec, StepSpec
from lookout.events import DetectionEvent, Frame
from lookout.frames import FrameBuffer
from lookout.scheduler import InferenceJob, Scheduler

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepPayload:
    """What the worker's `run` callable receives. Frames are already sliced;
    the model client only has to encode them."""

    kind: Capability
    prompt: str
    frames: tuple[Frame, ...]
    camera: str
    chain_id: str
    step_id: str
    generation: int
    center_ts: float
    timeout_s: float = 60.0  # the step's budget; the client uses it as its request timeout
    clip: bytes | None = None  # MP4 bytes for video_clip payloads; frames is empty then


@dataclass
class ChainRun:
    chain: ChainSpec
    trigger: DetectionEvent
    generation: int
    step: StepSpec
    center_ts: float
    due_at: float
    dispatched: bool = False
    deadline: float | None = None
    history: list[tuple[str, str]] = field(default_factory=list)  # (step_id, answer)


class Engine:
    def __init__(
        self,
        config: Config,
        scheduler: Scheduler,
        sink: ActionSink,
        buffers: dict[str, FrameBuffer],
        clock: Callable[[], float],
        trace: Callable[[str], None] | None = None,
        clips: ClipProvider | None = None,
    ) -> None:
        self.config = config
        self.scheduler = scheduler
        self.sink = sink
        self.buffers = buffers
        self.clock = clock
        self.clips = clips
        self._trace = trace or (lambda line: log.info("%s", line))
        self._runs: dict[str, ChainRun] = {}
        self._gen = itertools.count(1)
        self._lock = threading.RLock()

    # -- inputs ---------------------------------------------------------------

    def on_frame(self, frame: Frame) -> None:
        buffer = self.buffers.get(frame.camera)
        if buffer is not None:
            buffer.put(frame)

    def on_event(self, event: DetectionEvent) -> None:
        # A trigger's held_frames is the source's job (a gesture source emits an
        # event only once the pose has persisted that long), so by the time an
        # event reaches here the intent filter has already been applied.
        with self._lock:
            for chain in self.config.chains:
                if chain.camera != event.camera or not chain.trigger.matches(event.label):
                    continue
                if chain.id in self._runs:
                    self._trace(f"[{event.ts:7.2f}] {chain.id}: already active, ignoring {event.label}")
                    continue
                if not chain.steps:
                    assert chain.on_trigger is not None  # guaranteed by config validation
                    self._trace(f"[{event.ts:7.2f}] {chain.id}: {event.label} -> one-shot action, no escalation")
                    self.sink.fire(chain.on_trigger.action, self._context(chain, event))
                    continue
                run = ChainRun(
                    chain=chain, trigger=event, generation=next(self._gen),
                    step=chain.steps[0], center_ts=event.ts, due_at=event.ts,
                )
                self._runs[chain.id] = run
                self._trace(f"[{event.ts:7.2f}] {chain.id}: triggered by {event.label} ({event.confidence:.2f})")
                self._arm(run, chain.steps[0], center_ts=event.ts)
                self._maybe_dispatch(run)

    def on_result(self, job: InferenceJob, answer: Any) -> None:
        payload: StepPayload = job.payload
        with self._lock:
            run = self._runs.get(job.chain_id)
            if run is None or run.generation != payload.generation or run.step.id != job.step_id:
                self._trace(f"[{self.clock():7.2f}] {job.chain_id}: stale answer for {job.step_id} ignored")
                return
            text = str(answer).strip()
            run.history.append((run.step.id, text))
            key = match_outcome(text, run.step.outcomes)
            if key is None:
                self._trace(
                    f"[{self.clock():7.2f}] {run.chain.id}/{run.step.id}: unrecognised answer {text!r} "
                    f"(expected one of {sorted(run.step.outcomes)}), resetting chain"
                )
                self._finish(run)
                return
            outcome = run.step.outcomes[key]
            self._trace(f"[{self.clock():7.2f}] {run.chain.id}/{run.step.id}: {key}")
            self._apply(run, key, outcome)

    def on_error(self, job: InferenceJob, exc: BaseException) -> None:
        payload: StepPayload = job.payload
        with self._lock:
            run = self._runs.get(job.chain_id)
            if run is None or run.generation != payload.generation:
                return
            self._trace(f"[{self.clock():7.2f}] {run.chain.id}/{run.step.id}: inference failed ({exc}), resetting chain")
            self._finish(run)

    def tick(self) -> None:
        """Dispatch steps whose window has been seen; expire steps past their
        timeout. Call often (every frame is fine)."""
        now = self.clock()
        with self._lock:
            for run in list(self._runs.values()):
                if now - run.trigger.ts > run.chain.max_age_s:
                    dropped = self.scheduler.discard(run.chain.id)
                    self._trace(
                        f"[{now:7.2f}] {run.chain.id}/{run.step.id}: run is older than max_age_s="
                        f"{run.chain.max_age_s:g} ({dropped} queued job(s) dropped), giving up"
                    )
                    self._finish(run)
                elif not run.dispatched:
                    self._maybe_dispatch(run)
                elif run.deadline is not None and now > run.deadline:
                    dropped = self.scheduler.discard(run.chain.id)
                    self._trace(
                        f"[{now:7.2f}] {run.chain.id}/{run.step.id}: timed out after {run.step.timeout_s}s "
                        f"({dropped} queued job(s) dropped), resetting chain"
                    )
                    self._finish(run)

    # -- introspection --------------------------------------------------------

    @property
    def active(self) -> list[str]:
        with self._lock:
            return list(self._runs)

    def next_wakeup(self) -> float | None:
        """The earliest future instant at which tick() would do something:
        a step becoming due or a deadline expiring. None if nothing is armed."""
        with self._lock:
            times = [r.due_at for r in self._runs.values() if not r.dispatched]
            times += [r.deadline for r in self._runs.values() if r.dispatched and r.deadline is not None]
            times += [r.trigger.ts + r.chain.max_age_s for r in self._runs.values()]
            return min(times) if times else None

    # -- internals ------------------------------------------------------------

    def _arm(self, run: ChainRun, step: StepSpec, center_ts: float) -> None:
        window = run.chain.window_for(step) if step.payload != "image" else None
        run.step = step
        run.center_ts = center_ts
        run.due_at = center_ts + (window.after_s if window else 0.0)
        run.dispatched = False
        run.deadline = None

    def _maybe_dispatch(self, run: ChainRun) -> None:
        now = self.clock()
        if run.dispatched or now < run.due_at:
            return
        step, chain = run.step, run.chain
        buffer = self.buffers.get(chain.camera)
        frames: list[Frame] = []
        clip: bytes | None = None
        if step.payload == "video_clip":
            # Footage with its audio comes from the recorder, not the frame
            # buffer. Fetched here, on the tick thread: a few MB over the LAN.
            window = chain.window_for(step)
            assert window is not None
            start, end = run.center_ts - window.before_s, run.center_ts + window.after_s
            if self.clips is None:
                self._trace(f"[{now:7.2f}] {chain.id}/{step.id}: no clip provider, resetting chain")
                self._finish(run)
                return
            try:
                clip = self.clips.clip(chain.camera, start, end)
            except ClipError as exc:
                self._trace(f"[{now:7.2f}] {chain.id}/{step.id}: clip failed ({exc}), resetting chain")
                self._finish(run)
                return
            span = f"clip {start:.2f}..{end:.2f} ({len(clip) // 1024} KB)"
        elif buffer is not None and step.payload == "image":
            frame = buffer.at(run.center_ts)
            frames = [frame] if frame is not None else []
            span = f"{frames[0].ts:.2f}..{frames[-1].ts:.2f}" if frames else "no frames"
        elif buffer is not None:
            window = chain.window_for(step)
            assert window is not None
            frames = buffer.window(run.center_ts, window)
            span = f"{frames[0].ts:.2f}..{frames[-1].ts:.2f}" if frames else "no frames"
        else:
            span = "no frames"
        payload = StepPayload(
            kind=step.payload, prompt=step.prompt, frames=tuple(frames), camera=chain.camera,
            chain_id=chain.id, step_id=step.id, generation=run.generation, center_ts=run.center_ts,
            timeout_s=step.timeout_s, clip=clip,
        )
        self.scheduler.submit(
            InferenceJob(model=step.model, priority=step.priority, chain_id=chain.id, step_id=step.id, payload=payload)
        )
        run.dispatched = True
        run.deadline = now + step.timeout_s
        self._trace(
            f"[{now:7.2f}] {chain.id}/{step.id}: -> {step.model} p{step.priority} "
            f"{step.payload} x{len(frames)} [{span}]"
        )

    def _apply(self, run: ChainRun, key: str, outcome: OutcomeSpec) -> None:
        now = self.clock()
        if outcome.next is not None:
            self._arm(run, run.chain.step(outcome.next), center_ts=run.trigger.ts)
            self._maybe_dispatch(run)
        elif outcome.action is not None:
            self.sink.fire(outcome.action, self._context(run.chain, run.trigger, run.step.id, key))
            self._finish(run)
        elif outcome.retry_in_s is not None:
            step = run.step
            window = run.chain.window_for(step) if step.payload != "image" else None
            retry_at = now + outcome.retry_in_s
            center = retry_at - (window.after_s if window else 0.0)
            self._arm(run, step, center_ts=center)
            self._trace(f"[{now:7.2f}] {run.chain.id}/{step.id}: retry at {retry_at:.2f}")
        else:  # end
            self._finish(run)

    def _finish(self, run: ChainRun) -> None:
        self._runs.pop(run.chain.id, None)

    @staticmethod
    def _context(chain: ChainSpec, event: DetectionEvent, step_id: str | None = None, answer: str | None = None) -> ActionContext:
        return ActionContext(
            chain_id=chain.id, camera=event.camera, trigger_label=event.label,
            trigger_ts=event.ts, step_id=step_id, answer=answer,
        )


def match_outcome(answer: str, outcomes: dict[str, OutcomeSpec]) -> str | None:
    """Map a model's text to an outcome key. Exact first, then case-insensitive,
    then the first key appearing as a whole word (VLMs like to add prose around
    the label they were told to emit). None if nothing matches."""
    if answer in outcomes:
        return answer
    lowered = answer.lower()
    for key in outcomes:
        if key.lower() == lowered:
            return key
    for key in outcomes:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", answer, re.IGNORECASE):
            return key
    return None
