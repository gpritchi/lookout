"""Priority scheduler: the reason this project exists.

Inference capacity is the scarce resource. Several chains on several cameras all
want the same card, and a gesture command with a sub-second budget must not wait
behind a driveway classification with a 20-second one. This module decides who
goes next.

Design (settled — see CLAUDE.md):

- One priority queue per model name. Higher priority number wins; FIFO within a
  priority. An InferenceJob carries what the worker needs to run it and what the
  chain engine needs to route the answer back.
- A Worker is configured with the list of models it can serve. The live runner
  starts one per model: a worker blocked on a three-minute clip must not hold
  up one-second classifications for a model on another card. It repeatedly
  takes the highest-priority job across its models' queues and runs it. A
  second host is a worker with a different list — config, not redesign.
- No mid-inference preemption: a priority-100 job jumps every queue but never
  cancels an in-flight call.
- Every call the worker makes runs inside metrics.timed(model, tier).

Stdlib only: heapq under a Condition. Threads, not async — the worker blocks on
network calls and there are only ever a handful of them.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from lookout.metrics import timed

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceJob:
    """One request for a model's time.

    payload is opaque to the scheduler: the chain engine builds it (a frame ref,
    a list of frame refs, a prompt) and the worker's `run` callable knows how to
    turn it into a request. chain_id/step_id let the engine route the answer.
    """

    model: str
    priority: int
    chain_id: str
    step_id: str
    payload: Any = None
    enqueued_at: float = field(default_factory=time.monotonic)


class Scheduler:
    """Per-model priority queues with a single lock and condition.

    Ordering key is (-priority, sequence): highest priority first, then arrival
    order. The sequence counter is global across models, so "oldest across all my
    queues" is well defined when a worker compares queue heads.
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[tuple[int, int, InferenceJob]]] = {}
        self._seq = itertools.count()
        self._cond = threading.Condition()

    def submit(self, job: InferenceJob) -> None:
        with self._cond:
            heapq.heappush(
                self._queues.setdefault(job.model, []),
                (-job.priority, next(self._seq), job),
            )
            self._cond.notify_all()

    def next_job(self, models: Iterable[str], timeout: float | None = None) -> InferenceJob | None:
        """Pop the best job across the given models' queues, blocking up to
        `timeout` seconds for one to appear. Returns None on timeout. A model
        with no queue yet simply contributes nothing."""
        wanted = list(models)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                best_model = self._best_model_locked(wanted)
                if best_model is not None:
                    _, _, job = heapq.heappop(self._queues[best_model])
                    return job
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def _best_model_locked(self, models: list[str]) -> str | None:
        best: tuple[int, int] | None = None
        best_model = None
        for model in models:
            queue = self._queues.get(model)
            if not queue:
                continue
            key = queue[0][:2]
            if best is None or key < best:
                best, best_model = key, model
        return best_model

    def discard(self, chain_id: str) -> int:
        """Drop every queued job belonging to a chain. Used when a chain resets
        on timeout: its stale requests must not burn card time. Returns how many
        were dropped. Never touches an in-flight job (no preemption)."""
        dropped = 0
        with self._cond:
            for model, queue in self._queues.items():
                kept = [entry for entry in queue if entry[2].chain_id != chain_id]
                dropped += len(queue) - len(kept)
                heapq.heapify(kept)
                self._queues[model] = kept
        return dropped

    def pending(self, model: str | None = None) -> int:
        with self._cond:
            if model is not None:
                return len(self._queues.get(model, ()))
            return sum(len(q) for q in self._queues.values())


class Worker:
    """Drains the queues for its models, one job at a time.

    `run(job)` performs the inference and returns whatever the caller wants
    routed back (v1: the model's text answer). `on_result(job, result)` and
    `on_error(job, exc)` are invoked on the worker thread; keep them quick.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        models: Iterable[str],
        run: Callable[[InferenceJob], Any],
        on_result: Callable[[InferenceJob, Any], None],
        on_error: Callable[[InferenceJob, BaseException], None] | None = None,
        tier: str = "tier2",
        poll_s: float = 0.25,
    ) -> None:
        self.scheduler = scheduler
        self.models = list(models)
        self._run = run
        self._on_result = on_result
        self._on_error = on_error
        self.tier = tier
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self, timeout: float | None = 0) -> bool:
        """Take and run at most one job. Returns whether one was run. Used by
        tests and by anyone who wants to drive the worker synchronously."""
        job = self.scheduler.next_job(self.models, timeout=timeout)
        if job is None:
            return False
        try:
            with timed(job.model, self.tier):
                result = self._run(job)
        except Exception as exc:  # noqa: BLE001 — one bad call must not kill the worker
            log.exception("inference failed: chain=%s step=%s model=%s", job.chain_id, job.step_id, job.model)
            if self._on_error is not None:
                self._on_error(job, exc)
            return True
        self._on_result(job, result)
        return True

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("worker already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"worker[{','.join(self.models)}]", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(join_timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.run_once(timeout=self._poll_s)
