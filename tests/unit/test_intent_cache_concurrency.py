"""The intent cold-load caches: two concurrent cold requests load once.

Since #657 ``recognize_intent`` runs in worker threads, so the index and the
encoder caches can be hit cold from two threads at once. The loader here
blocks until released; without the lock a second thread enters it too.
Every wait is bounded, so a regression fails rather than hangs.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from src.internal.configs import AppSettings
from src.internal.servers.web.intent import similarity
from src.model.pre_training.intents import model as model_mod

_WAIT = 5.0
# How long to give a second thread to (wrongly) enter the loader.
_SECOND_ENTRY_GRACE = 0.5


class _BlockingLoader:
    """Counts calls; the first blocks until ``release`` is set."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.calls = 0
        self._lock = threading.Lock()
        self.entered = threading.Event()
        self.second_entered = threading.Event()
        self.release = threading.Event()
        self._result = result
        self._error = error

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
            if self.calls == 1:
                self.entered.set()
            else:
                self.second_entered.set()
        assert self.release.wait(_WAIT), "loader never released"
        if self._error is not None:
            raise self._error
        return self._result if self._result is not None else object()


def _race(target, loader):
    """Run *target* in two threads with the loader held; return outcomes."""
    outcomes: list[object] = [None, None]

    def run(i):
        try:
            outcomes[i] = target()
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            outcomes[i] = exc

    first = threading.Thread(target=run, args=(0,), daemon=True)
    second = threading.Thread(target=run, args=(1,), daemon=True)
    first.start()
    assert loader.entered.wait(_WAIT), "first thread never reached the loader"
    second.start()
    loader.second_entered.wait(_SECOND_ENTRY_GRACE)
    loader.release.set()
    first.join(_WAIT)
    second.join(_WAIT)
    assert not first.is_alive() and not second.is_alive()
    return outcomes


class _FakeIndex:
    encoder = model_mod.DEFAULT_ENCODER

    def low_support_modules(self):
        return []


def test_concurrent_cold_index_load_loads_once(tmp_path, monkeypatch):
    monkeypatch.setattr(similarity, "_INTENT_INDEXES", {})
    index = _FakeIndex()
    loader = _BlockingLoader(result=index)
    monkeypatch.setattr(model_mod, "IntentIndex", types.SimpleNamespace(load=loader))
    settings = AppSettings(intent_index_path=tmp_path)

    outcomes = _race(lambda: similarity.load_intent_index(settings), loader)

    assert loader.calls == 1
    assert outcomes == [index, index]


def _install_fake_sentence_transformers(monkeypatch, loader):
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = loader
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setattr(model_mod, "_MODEL_CACHE", {})


def test_concurrent_cold_encoder_load_loads_once(monkeypatch):
    encoder = object()
    loader = _BlockingLoader(result=encoder)
    _install_fake_sentence_transformers(monkeypatch, loader)

    outcomes = _race(lambda: model_mod._model("some/encoder"), loader)

    assert loader.calls == 1
    assert outcomes == [encoder, encoder]


def test_concurrent_cold_load_failure_is_decided_once(monkeypatch):
    boom = OSError("download failed")
    loader = _BlockingLoader(error=boom)
    _install_fake_sentence_transformers(monkeypatch, loader)

    outcomes = _race(lambda: model_mod._model("some/encoder"), loader)

    assert loader.calls == 1
    assert outcomes[0] is boom
    assert isinstance(outcomes[1], RuntimeError)
    assert outcomes[1].__cause__ is boom
    with pytest.raises(RuntimeError, match="not retrying"):
        model_mod._model("some/encoder")
