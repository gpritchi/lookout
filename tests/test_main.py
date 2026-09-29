"""The CLI's process-level behaviour."""

import os
import signal
import time

import pytest

from lookout.__main__ import stop_on_sigterm


def test_sigterm_becomes_keyboard_interrupt():
    """In a pod lookout is PID 1, which ignores an unhandled SIGTERM. The
    handler turns it into the KeyboardInterrupt run_live already stops on."""
    previous = signal.getsignal(signal.SIGTERM)
    try:
        stop_on_sigterm()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)  # the handler runs at the next bytecode; never reached
    finally:
        signal.signal(signal.SIGTERM, previous)
