"""`serving_cache()` is off until the web lifespan configures it."""

from __future__ import annotations

import pytest

from src.internal.cache import serving


@pytest.fixture(autouse=True)
def _reset():
    serving.reset_serving_cache()
    yield
    serving.reset_serving_cache()


def test_off_until_configured():
    assert serving.serving_cache() is None


def test_zero_ttl_leaves_it_off():
    serving.configure_serving_cache(0)
    assert serving.serving_cache() is None


def test_configure_then_reset():
    serving.configure_serving_cache(30)
    cache = serving.serving_cache()
    assert cache is not None
    cache.set("k", 1)
    assert serving.serving_cache() is cache
    serving.reset_serving_cache()
    assert serving.serving_cache() is None


def test_reconfigure_replaces_the_instance():
    serving.configure_serving_cache(30)
    first = serving.serving_cache()
    first.set("k", 1)
    serving.configure_serving_cache(30)
    assert serving.serving_cache() is not first
    assert serving.serving_cache().get("k") is None
