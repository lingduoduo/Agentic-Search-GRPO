"""AGENTIC_SEARCH_EMBEDDED_RETRIEVAL points the retrieval URL at the API itself.

The static demo corpus does not need its own process: with the flag set, the web
app serves /retrieval/retrieve in-process, and every consumer that reads
``services.retrieval_url`` must reach it without being told the URL separately.
"""

from __future__ import annotations

from src.internal.configs import load_app_settings
from src.internal.configs.app_configs import DEFAULT_RETRIEVAL_URL


def test_unset_keeps_the_standalone_server_default():
    services = load_app_settings({}).services

    assert services.embedded_retrieval is None
    assert services.retrieval_url == DEFAULT_RETRIEVAL_URL


def test_embedded_points_retrieval_at_the_api_on_its_documented_port():
    services = load_app_settings({"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "demo"}).services

    assert services.embedded_retrieval == "demo"
    assert services.retrieval_url == "http://127.0.0.1:7860/retrieval/retrieve"


def test_embedded_follows_an_explicit_web_port():
    services = load_app_settings(
        {"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "demo", "AGENTIC_SEARCH_WEB_PORT": "9000"}
    ).services

    assert services.retrieval_url == "http://127.0.0.1:9000/retrieval/retrieve"


def test_an_explicit_retrieval_url_still_wins():
    services = load_app_settings(
        {
            "AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "demo",
            "AGENTIC_SEARCH_RETRIEVAL_URL": "http://search.test/retrieve",
        }
    ).services

    assert services.retrieval_url == "http://search.test/retrieve"
