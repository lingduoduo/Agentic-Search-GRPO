import os
from pathlib import Path

import pytest

# The web app's lifespan loads SEARCH_AGENT_MODEL onto the device on every
# TestClient startup — turning a fast web test file into a multi-minute (or,
# offline, hanging) run. No test needs the real model: they inject a mock
# manager or exercise the no-model 400 path. Neutralize the vars for the whole
# session so the default `pytest` run stays fast. Empty-but-set wins over .env
# because create_web_app() calls load_dotenv(override=False), and get_env_str()
# treats "" as unset. Mirrors examples/run_web_integration_tests.sh.
os.environ["SEARCH_AGENT_MODEL"] = ""
os.environ["SEARCH_AGENT_SERVER_URL"] = ""
os.environ["AGENTIC_SEARCH_SEARCH_DIRECT_SEMANTIC"] = "0"
# Same mechanism for the remote LLM: a developer's .env key would make
# create_web_app() build a real client, and with /api/agent summarization on
# by default a long-history test would send fixture text to the paid provider.
# Tests that need a key set their own with monkeypatch.
os.environ["OPENAI_API_KEY"] = ""
os.environ["GEN_AI_API_KEY"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_TOP_LEVEL_COLLECTION_IGNORE_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".understand-anything",
    "agentic_search.egg-info",
    "data",
    "dist",
    "models",
}


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # noqa: ARG001
    try:
        relative = collection_path.relative_to(config.rootpath)
    except ValueError:
        return False
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in _TOP_LEVEL_COLLECTION_IGNORE_DIRS:
        return True
    return len(parts) >= 2 and parts[:2] in {
        ("web", "dist"),
        ("web", "node_modules"),
    }


@pytest.fixture(autouse=True)
def _fresh_timeout_policies():
    """Policies are cached per process; tests that set env vars must not leak."""
    from src.internal.configs.timeouts import reset_timeout_policies

    reset_timeout_policies()
    yield
    reset_timeout_policies()


@pytest.fixture(autouse=True)
def _fresh_circuit_breakers():
    """Breakers are process-wide; one test's failures must not open another's."""
    from src.internal.resilience.circuit_breaker import reset_breakers

    reset_breakers()
    yield
    reset_breakers()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "load: marks tests as load/performance tests")
    config.addinivalue_line(
        "markers", "integration: marks tests requiring a live server stack"
    )
    config.addinivalue_line(
        "markers", "alembic: marks migration tests that exercise Alembic flows"
    )
