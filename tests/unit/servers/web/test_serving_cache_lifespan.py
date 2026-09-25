"""The web app's lifespan owns the serving cache: it configures it from the
settings on startup and resets it on shutdown, so a process that never starts
the web app caches nothing, and a test app leaves nothing behind."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.internal.cache import serving
from src.internal.configs.app_configs import load_app_settings
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


@pytest.fixture(autouse=True)
def _reset():
    serving.reset_serving_cache()
    yield
    serving.reset_serving_cache()


def test_env_reaches_the_experience_settings():
    settings = load_app_settings({"AGENTIC_SEARCH_SEARCH_CACHE_TTL": "0"})
    assert settings.services.search_cache_ttl_seconds == 0
    assert SearchExperienceSettings.from_app_settings(settings).search_cache_ttl == 0


def test_default_ttl_is_five_minutes():
    settings = load_app_settings({})
    assert settings.services.search_cache_ttl_seconds == 300
    assert SearchExperienceSettings.from_app_settings(settings).search_cache_ttl == 300


def test_lifespan_configures_then_resets(tmp_path):
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "a.sqlite3", search_cache_ttl=42)
    )
    assert serving.serving_cache() is None
    with TestClient(app):
        cache = serving.serving_cache()
        assert cache is not None
        cache.set("k", 1)
        assert cache.get("k") == 1
    assert serving.serving_cache() is None


def test_zero_ttl_never_configures(tmp_path):
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "b.sqlite3", search_cache_ttl=0)
    )
    with TestClient(app):
        assert serving.serving_cache() is None


def test_stale_window_defaults_to_an_hour():
    settings = load_app_settings({})
    assert settings.services.search_cache_stale_seconds == 3600
    experience = SearchExperienceSettings.from_app_settings(settings)
    assert experience.search_cache_stale == 3600


def test_stale_window_env_is_parsed_and_zero_is_off():
    settings = load_app_settings({"AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS": "0"})
    assert settings.services.search_cache_stale_seconds == 0
    experience = SearchExperienceSettings.from_app_settings(settings)
    assert experience.search_cache_stale == 0


@pytest.mark.parametrize("value", ["soon", "-5"])
def test_bad_stale_window_is_rejected(value):
    with pytest.raises(ValueError, match="AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS"):
        load_app_settings({"AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS": value})


def test_lifespan_passes_the_stale_window(tmp_path):
    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "c.sqlite3", search_cache_ttl=42, search_cache_stale=900
        )
    )
    with TestClient(app):
        assert serving.serving_cache()._stale == 900.0
