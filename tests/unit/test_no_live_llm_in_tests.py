"""The unit suite must never build a real remote LLM client.

create_web_app() calls load_dotenv(), so a developer's .env OPENAI_API_KEY /
GEN_AI_API_KEY would otherwise build an OpenAICompatibleLLM -- and with
/api/agent summarization on by default, a long-history test then sends
fixture text to the paid provider. tests/conftest.py blanks both keys.
"""

from __future__ import annotations

from src.internal.servers.web import app as web_app
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def test_default_web_app_builds_no_real_llm_client(monkeypatch, tmp_path):
    built: list = []

    class _Recorder:
        def __init__(self, *args, **kwargs):
            # A marker, never the config: a failure must not print the key.
            built.append("OpenAICompatibleLLM")

    monkeypatch.setattr(web_app, "OpenAICompatibleLLM", _Recorder)

    create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))

    assert built == []
