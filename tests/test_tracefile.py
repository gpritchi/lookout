"""The trace file a long-running deployment keeps: every line stamped with UTC
wall-clock time (engine time restarts at 0 with the process), appended across
restarts, and a write failure never takes detection down."""

import re
from datetime import datetime, timezone

import pytest

from lookout.runtime import TraceFile

STAMP = re.compile(r"^(\S+) (.*)$")


def lines(path) -> list[tuple[datetime, str]]:
    out = []
    for raw in path.read_text().splitlines():
        stamp, rest = STAMP.match(raw).groups()
        out.append((datetime.fromisoformat(stamp), rest))
    return out


def test_lines_are_stamped_with_utc_wall_clock(tmp_path):
    path = tmp_path / "trace.log"
    now = datetime.now(timezone.utc)
    before = now.replace(microsecond=now.microsecond // 1000 * 1000)  # stamps are to the millisecond
    with TraceFile(path, started="lookout run --config c.json") as trace:
        trace("[   3.06] driveway-arrivals: triggered by car (0.74)")
    after = datetime.now(timezone.utc)
    got = lines(path)
    assert [text for _, text in got] == [
        "--- started: lookout run --config c.json",
        "[   3.06] driveway-arrivals: triggered by car (0.74)",
    ]
    for stamp, _ in got:
        assert stamp.tzinfo is not None and before <= stamp <= after


def test_appends_across_restarts(tmp_path):
    path = tmp_path / "trace.log"
    with TraceFile(path, started="first") as trace:
        trace("one")
    with TraceFile(path, started="second") as trace:
        trace("two")
    assert [text for _, text in lines(path)] == ["--- started: first", "one", "--- started: second", "two"]


def test_unopenable_path_fails_at_startup(tmp_path):
    with pytest.raises(OSError):
        TraceFile(tmp_path / "missing-dir" / "trace.log", started="x")


def test_write_failure_is_reported_once_and_never_raises(tmp_path, caplog):
    trace = TraceFile(tmp_path / "trace.log", started="x")
    trace._fh.close()  # what a stale NFS handle looks like from here: every write fails
    trace("a")
    trace("b")
    assert trace.failed == 2
    assert sum("trace file write failed" in r.getMessage() for r in caplog.records) == 1
