"""Scheduler basics, per CLAUDE.md: higher priority first, FIFO within equal
priority, a worker never takes another model's job. Plus the worker loop itself
and the timing wrapper it runs every call inside."""

import threading

from prometheus_client import REGISTRY

from lookout.scheduler import InferenceJob, Scheduler, Worker


def job(model: str, priority: int, step: str = "s", chain: str = "c") -> InferenceJob:
    return InferenceJob(model=model, priority=priority, chain_id=chain, step_id=step)


def drain(scheduler: Scheduler, models: list[str]) -> list[str]:
    out = []
    while (j := scheduler.next_job(models, timeout=0)) is not None:
        out.append(j.step_id)
    return out


def test_higher_priority_dequeues_first():
    s = Scheduler()
    s.submit(job("vlm", 10, "low"))
    s.submit(job("vlm", 100, "gesture"))
    s.submit(job("vlm", 50, "mid"))
    assert drain(s, ["vlm"]) == ["gesture", "mid", "low"]


def test_fifo_within_equal_priority():
    s = Scheduler()
    for name in ("a", "b", "c"):
        s.submit(job("vlm", 50, name))
    assert drain(s, ["vlm"]) == ["a", "b", "c"]


def test_worker_never_takes_another_models_job():
    s = Scheduler()
    s.submit(job("omni", 100, "clip"))
    s.submit(job("vlm", 1, "image"))
    assert s.next_job(["vlm"], timeout=0).step_id == "image"
    assert s.next_job(["vlm"], timeout=0) is None
    assert s.pending("omni") == 1


def test_best_across_multiple_queues():
    s = Scheduler()
    s.submit(job("vlm", 50, "vlm-first"))
    s.submit(job("omni", 60, "omni-high"))
    s.submit(job("vlm", 60, "vlm-high-later"))
    # equal priority across models resolves by arrival order
    assert drain(s, ["vlm", "omni"]) == ["omni-high", "vlm-high-later", "vlm-first"]


def test_unknown_model_is_empty_not_error():
    s = Scheduler()
    assert s.next_job(["never-seen"], timeout=0) is None
    assert s.pending() == 0


def test_discard_drops_only_that_chain():
    s = Scheduler()
    s.submit(job("vlm", 50, "keep", chain="A"))
    s.submit(job("vlm", 90, "stale", chain="B"))
    s.submit(job("omni", 10, "stale2", chain="B"))
    assert s.discard("B") == 2
    assert drain(s, ["vlm", "omni"]) == ["keep"]


def test_blocking_wait_wakes_on_submit():
    s = Scheduler()
    got: list[InferenceJob] = []

    def waiter():
        got.append(s.next_job(["vlm"], timeout=2.0))

    t = threading.Thread(target=waiter)
    t.start()
    s.submit(job("vlm", 1, "late"))
    t.join(2.0)
    assert got and got[0].step_id == "late"


def _count(model: str, tier: str) -> float:
    return REGISTRY.get_sample_value("lookout_inference_seconds_count", {"model": model, "tier": tier}) or 0.0


def test_worker_runs_job_and_routes_result():
    s = Scheduler()
    results = []
    w = Worker(s, ["vlm"], run=lambda j: f"answer:{j.step_id}", on_result=lambda j, r: results.append(r))
    assert w.run_once() is False  # nothing queued
    before = _count("vlm", "tier2")
    s.submit(job("vlm", 5, "classify"))
    assert w.run_once() is True
    assert results == ["answer:classify"]
    assert _count("vlm", "tier2") == before + 1  # the call was timed


def test_worker_error_goes_to_on_error_and_worker_survives():
    s = Scheduler()
    errors, results = [], []

    def run(j):
        if j.step_id == "boom":
            raise RuntimeError("endpoint down")
        return "ok"

    w = Worker(s, ["vlm"], run=run, on_result=lambda j, r: results.append(r), on_error=lambda j, e: errors.append(str(e)))
    s.submit(job("vlm", 9, "boom"))
    s.submit(job("vlm", 1, "fine"))
    assert w.run_once() and w.run_once()
    assert errors == ["endpoint down"]
    assert results == ["ok"]


def test_slow_model_does_not_block_a_fast_one_with_a_worker_each():
    """The live finding: one worker over two models let a long clip call
    starve quick classifications. A worker per model keeps them independent."""
    s = Scheduler()
    release = threading.Event()
    fast_done = threading.Event()
    seen: list[str] = []

    def run(j):
        if j.model == "slow":
            release.wait(5.0)  # a three-minute clip, in miniature
        return j.step_id

    def on_result(j, r):
        seen.append(r)
        if j.model == "fast":
            fast_done.set()

    workers = [Worker(s, [m], run=run, on_result=on_result, poll_s=0.01) for m in ("slow", "fast")]
    for w in workers:
        w.start()
    try:
        s.submit(job("slow", 40, "clip"))
        s.submit(job("fast", 50, "classify"))
        assert fast_done.wait(2.0), "fast model's job waited behind the slow model's"
        assert seen == ["classify"]
        release.set()
    finally:
        for w in workers:
            w.stop()
    assert sorted(seen) == ["classify", "clip"]


def test_worker_thread_drains_in_priority_order():
    s = Scheduler()
    done = threading.Event()
    seen: list[str] = []

    def on_result(j, r):
        seen.append(j.step_id)
        if len(seen) == 3:
            done.set()

    # Queue everything before starting so ordering is deterministic.
    s.submit(job("vlm", 10, "low"))
    s.submit(job("vlm", 100, "gesture"))
    s.submit(job("vlm", 50, "mid"))
    w = Worker(s, ["vlm"], run=lambda j: None, on_result=on_result, poll_s=0.01)
    w.start()
    try:
        assert done.wait(2.0)
    finally:
        w.stop()
    assert seen == ["gesture", "mid", "low"]
