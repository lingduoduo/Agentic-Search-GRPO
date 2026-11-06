"""Reset memory lifecycle module state between tests.

The per-user guard set and the after_turn task set are module-level, so a test
that leaks either can make a later test wait on a guard or hold a dead task.
"""

from __future__ import annotations

import contextlib

import pytest

from src.internal.hooks import lifecycle as hooks_lifecycle
from src.internal.memory import lifecycle as memory_lifecycle


@pytest.fixture(autouse=True)
def _reset_memory_lifecycle_state():
    yield
    with memory_lifecycle._busy_lock:
        memory_lifecycle._busy.clear()
    for task in list(hooks_lifecycle._tasks):
        with contextlib.suppress(RuntimeError):  # its loop may already be closed
            task.cancel()
    hooks_lifecycle._tasks.clear()
