"""Wiring: config + source + scheduler + worker + engine + sink, and the loop
that drives them.

Two clocks. VirtualClock for replay: time is whatever the stream says it is,
and the loop jumps straight to the next instant anything needs to happen, so a
minute of footage replays in milliseconds and every test is deterministic.
The wall clock for live runs, where the same loop sleeps instead of jumping.

ScriptedModel stands in for the VLM here: it answers each step from a JSON
script, in order, repeating the last answer. The real client plugs into the same
Worker slot in a later PR.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from lookout.actions import ActionSink, MockSink
from lookout.config import Config
from lookout.engine import Engine, StepPayload
from lookout.events import DetectionEvent, Frame, Source
from lookout.frames import FrameBuffer
from lookout.scheduler import InferenceJob, Scheduler, Worker

log = logging.getLogger(__name__)


class VirtualClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance_to(self, ts: float) -> None:
        if ts > self.t:
            self.t = ts


class ScriptedModel:
    """Answers per step id from a script: {"classify-vehicle": ["GREY_VAN"], ...}.
    Consumes answers in order and repeats the last one. Unknown steps get
    `default`, which mimics a model going off-script."""

    def __init__(self, script: dict[str, list[str]], default: str = "UNSCRIPTED") -> None:
        self._script = {k: list(v) for k, v in script.items()}
        self._default = default
        self.calls: list[StepPayload] = []

    @classmethod
    def from_file(cls, path: str | Path) -> "ScriptedModel":
        with Path(path).open() as fh:
            return cls(json.load(fh))

    def __call__(self, job: InferenceJob) -> str:
        payload: StepPayload = job.payload
        self.calls.append(payload)
        answers = self._script.get(job.step_id)
        if not answers:
            return self._default
        return answers.pop(0) if len(answers) > 1 else answers[0]


@dataclass
class RunReport:
    trace: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    inference_calls: int = 0
    stream_end_ts: float = 0.0
    final_ts: float = 0.0


def build_buffers(config: Config) -> dict[str, FrameBuffer]:
    """One ring buffer per camera, deep enough for the widest look-back any of
    its chains asks for, plus the longest a chain can spend in flight before a
    later step samples around the original trigger (every step's timeout, and
    the widest after_s wait)."""
    buffers: dict[str, FrameBuffer] = {}
    for name in config.cameras:
        lookback = config.buffer_seconds(name)
        in_flight = 0.0
        for chain in config.chains:
            if chain.camera != name:
                continue
            windows = [chain.window_for(s) for s in chain.steps]
            longest_wait = max((w.after_s for w in windows if w is not None), default=0.0)
            in_flight = max(in_flight, sum(s.timeout_s for s in chain.steps) + longest_wait)
        buffers[name] = FrameBuffer(keep_s=lookback + in_flight)
    return buffers


def run_replay(
    config: Config,
    source: Source,
    model: Callable[[InferenceJob], str],
    sink: ActionSink | None = None,
    idle_horizon_s: float = 600.0,
) -> RunReport:
    """Drive the whole pipeline on a virtual clock until the stream ends and
    every armed chain has resolved (or `idle_horizon_s` passes with nothing
    left to do). Inference runs inline on the loop thread so the result is
    deterministic."""
    clock = VirtualClock()
    sink = sink if sink is not None else MockSink()
    report = RunReport()
    scheduler = Scheduler()
    buffers = build_buffers(config)

    def trace(line: str) -> None:
        report.trace.append(line)
        log.info("%s", line)

    engine = Engine(config, scheduler, sink, buffers, clock.now, trace=trace)

    def on_result(job: InferenceJob, answer: str) -> None:
        report.inference_calls += 1
        engine.on_result(job, answer)

    worker = Worker(scheduler, list(config.models), run=model, on_result=on_result, on_error=engine.on_error)

    def drain() -> None:
        while worker.run_once(timeout=0):
            pass

    def settle_until(ts: float) -> None:
        """Advance the clock to `ts`, stopping at every instant strictly before
        it that the engine wants a tick. A wake-up at exactly `ts` waits until
        the item at `ts` (a frame, usually) has been stored, so a window ending
        at `ts` includes that frame."""
        while True:
            wake = engine.next_wakeup()
            if wake is None or wake >= ts:
                break
            clock.advance_to(wake)
            engine.tick()
            drain()
        clock.advance_to(ts)

    for item in source.stream():
        settle_until(item.ts)
        if isinstance(item, Frame):
            engine.on_frame(item)
        elif isinstance(item, DetectionEvent):
            engine.on_event(item)
            drain()
        engine.tick()
        drain()
    report.stream_end_ts = clock.now()

    # Stream is over; let armed chains time out or resolve.
    while engine.active or scheduler.pending():
        wake = engine.next_wakeup()
        if wake is None or wake > report.stream_end_ts + idle_horizon_s:
            break
        clock.advance_to(wake)
        engine.tick()
        drain()
    report.final_ts = clock.now()

    if isinstance(sink, MockSink):
        for action, ctx in sink.fired:
            extras = ", ".join(f"{k}={v}" for k, v in action.model_dump(exclude={"type"}).items())
            report.actions.append(f"{action.type}({extras}) <- {ctx.chain_id} on {ctx.camera}")
    return report


def run_live(
    config: Config,
    sources: dict[str, Source],
    model: Callable[[InferenceJob], str],
    sink: ActionSink | None = None,
    max_seconds: float = 0.0,
    stop_after_actions: int = 0,
    trace: Callable[[str], None] | None = None,
) -> RunReport:
    """Wall-clock run against live sources. Each source streams on its own
    thread into the engine; the worker runs inference on another; the main
    thread ticks the engine until a stop condition: `max_seconds` elapsed,
    `stop_after_actions` actions fired (a test that wants "the first delivery"
    and nothing more), or Ctrl-C.

    Timestamps are the sources' own (wall clock for RTSP), so the engine's
    clock is the same wall clock offset to start at 0, and every source must
    use that same clock — VideoSource does when given `clock`.
    """
    sink = sink if sink is not None else MockSink()
    report = RunReport()
    scheduler = Scheduler()
    buffers = build_buffers(config)
    start = time.monotonic()

    def clock() -> float:
        return time.monotonic() - start

    def _trace(line: str) -> None:
        report.trace.append(line)
        (trace or (lambda s: log.info("%s", s)))(line)

    engine = Engine(config, scheduler, sink, buffers, clock, trace=_trace)

    def on_result(job: InferenceJob, answer: str) -> None:
        report.inference_calls += 1
        engine.on_result(job, answer)

    worker = Worker(scheduler, list(config.models), run=model, on_result=on_result, on_error=engine.on_error)
    stop = threading.Event()

    def pump(name: str, source: Source) -> None:
        try:
            for item in source.stream():
                if stop.is_set():
                    break
                if isinstance(item, Frame):
                    engine.on_frame(item)
                else:
                    engine.on_event(item)
        except Exception:  # noqa: BLE001 — a dead camera must not take the loop down silently
            log.exception("source %s stopped", name)
        finally:
            _trace(f"[{clock():7.2f}] source {name}: stream ended")

    threads = [threading.Thread(target=pump, args=(n, s), name=f"source[{n}]", daemon=True) for n, s in sources.items()]
    worker.start()
    for t in threads:
        t.start()
    try:
        while not stop.is_set():
            time.sleep(0.25)
            engine.tick()
            fired = len(sink.fired) if isinstance(sink, MockSink) else 0
            if stop_after_actions and fired >= stop_after_actions:
                _trace(f"[{clock():7.2f}] stopping: {fired} action(s) fired")
                break
            if max_seconds and clock() >= max_seconds:
                _trace(f"[{clock():7.2f}] stopping: {max_seconds:.0f}s elapsed")
                break
            if all(not t.is_alive() for t in threads) and not engine.active and not scheduler.pending():
                break
    except KeyboardInterrupt:
        _trace(f"[{clock():7.2f}] stopping: interrupted")
    finally:
        stop.set()
        worker.stop()
    report.final_ts = clock()
    if isinstance(sink, MockSink):
        for action, ctx in sink.fired:
            extras = ", ".join(f"{k}={v}" for k, v in action.model_dump(exclude={"type"}).items())
            report.actions.append(f"{action.type}({extras}) <- {ctx.chain_id} on {ctx.camera}")
    return report
