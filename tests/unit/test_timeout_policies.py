import dataclasses
import math

import pytest

from src.internal.configs.timeouts import (
    TIMEOUTS_PATH_ENV,
    get_timeout_policies,
    load_timeout_policies,
    reset_timeout_policies,
    use_timeout_policies,
)

TODAY = {
    "tools": {
        "public_data": {
            "timeout_seconds": 10.0,
            "max_attempts": 3,
            "backoff_seconds": (0.4, 0.8),
            "retry_budget_seconds": 15.0,
            "overpass_timeout_seconds": 30.0,
            "overpass_query_timeout_seconds": 25.0,
        },
        "web_search": {
            "google_timeout_seconds": 15.0,
            "serpapi_timeout_seconds": 15.0,
            "serper_timeout_seconds": 10.0,
            "fetch_page_timeout_seconds": 15.0,
            "fetch_pages_timeout_seconds": 10.0,
            "query_timeout_seconds": 15.0,
        },
        "search_router": {"timeout_seconds": 15.0, "max_retries": 3},
        "openapi": {"timeout_seconds": 15.0},
        "mcp": {"timeout_seconds": 30.0, "sse_read_timeout_seconds": 300.0},
    },
    "retrieval": {
        "client": {
            "timeout_seconds": 10.0,
            "max_retries": 3,
            "backoff_base_seconds": 0.5,
        },
        "search_runner": {"timeout_seconds": 15.0, "max_retries": 3},
        "web_hybrid": {
            "provider_timeout_seconds": 5.0,
            "provider_max_retries": 1,
            "provider_wait_seconds": 8.0,
        },
    },
    "rerank": {"timeout_seconds": 10.0},
    "llm": {
        "socket_read_timeout_seconds": 120.0,
        "remote_total_timeout_seconds": 120.0,
        "local_generation_timeout_seconds": 120.0,
        "local_heartbeat_seconds": 10.0,
        "sufficiency_timeout_seconds": 5.0,
        "grounded_max_retries": 1,
    },
    "tool_loop": {
        "approval_timeout_seconds": 60.0,
        "escalation_timeout_seconds": 120.0,
        "max_escalations": 3,
        "tool_evidence_timeout_seconds": 5.0,
        "recovery": {
            "max_retries": 2,
            "backoff_seconds": (0.5, 1.0),
            "retry_after_cap_seconds": 4.0,
            "retry_budget_seconds": 10.0,
        },
    },
    "sse": {"heartbeat_seconds": 15.0},
    "circuit_breaker": {"failure_threshold": 5, "open_seconds": 30.0},
    "readiness": {"probe_timeout_seconds": 2.0},
}


def test_bundled_defaults_are_todays_values():
    assert dataclasses.asdict(load_timeout_policies({})) == TODAY


def test_bundled_file_loads_through_package_resources():
    from importlib import resources

    assert resources.files("src.internal.configs").joinpath("timeouts.toml").is_file()


def test_partial_override_file_deep_merges(tmp_path):
    f = tmp_path / "t.toml"
    f.write_text("[llm]\nsocket_read_timeout_seconds = 30\n")
    p = load_timeout_policies({TIMEOUTS_PATH_ENV: str(f)})
    assert p.llm.socket_read_timeout_seconds == 30.0
    assert p.llm.remote_total_timeout_seconds == 120.0
    assert p.tools.public_data.max_attempts == 3


def test_unknown_key_names_its_dotted_path(tmp_path):
    f = tmp_path / "t.toml"
    f.write_text("[llm]\nsocket_read_timout_seconds = 30\n")
    with pytest.raises(ValueError, match=r"llm\.socket_read_timout_seconds"):
        load_timeout_policies({TIMEOUTS_PATH_ENV: str(f)})


def test_unknown_table_names_its_path():
    with pytest.raises(ValueError, match=r"'nope'"):
        load_timeout_policies({}, overrides={"nope": {"x": 1}})


def test_missing_override_file_errors(tmp_path):
    with pytest.raises(ValueError, match="AGENTIC_SEARCH_TIMEOUTS_PATH"):
        load_timeout_policies({TIMEOUTS_PATH_ENV: str(tmp_path / "absent.toml")})


def test_unparsable_override_file_errors(tmp_path):
    f = tmp_path / "t.toml"
    f.write_text("[llm\n")
    with pytest.raises(ValueError, match="invalid TOML"):
        load_timeout_policies({TIMEOUTS_PATH_ENV: str(f)})


def test_empty_path_env_means_bundled_only():
    assert load_timeout_policies({TIMEOUTS_PATH_ENV: "  "}) == load_timeout_policies({})


@pytest.mark.parametrize(
    "overrides, path",
    [
        ({"rerank": {"timeout_seconds": 0}}, "rerank.timeout_seconds"),
        ({"rerank": {"timeout_seconds": -1}}, "rerank.timeout_seconds"),
        ({"rerank": {"timeout_seconds": True}}, "rerank.timeout_seconds"),
        ({"rerank": {"timeout_seconds": "10"}}, "rerank.timeout_seconds"),
        ({"rerank": {"timeout_seconds": math.inf}}, "rerank.timeout_seconds"),
        ({"retrieval": {"client": {"max_retries": 0}}}, "retrieval.client.max_retries"),
        (
            {"retrieval": {"client": {"max_retries": 2.5}}},
            "retrieval.client.max_retries",
        ),
        (
            {"tools": {"public_data": {"max_attempts": 0}}},
            "tools.public_data.max_attempts",
        ),
        ({"tool_loop": {"max_escalations": 0}}, "tool_loop.max_escalations"),
        (
            {"tool_loop": {"recovery": {"max_retries": -1}}},
            "tool_loop.recovery.max_retries",
        ),
        (
            {"tool_loop": {"recovery": {"backoff_seconds": []}}},
            "tool_loop.recovery.backoff_seconds",
        ),
        (
            {"tool_loop": {"recovery": {"backoff_seconds": [0.5, -1]}}},
            "tool_loop.recovery.backoff_seconds",
        ),
        ({"llm": 5}, "llm"),
        (
            {"circuit_breaker": {"failure_threshold": 0}},
            "circuit_breaker.failure_threshold",
        ),
        ({"circuit_breaker": {"open_seconds": 0}}, "circuit_breaker.open_seconds"),
    ],
)
def test_invalid_values_error_naming_the_key(overrides, path):
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        load_timeout_policies({}, overrides=overrides)


def test_zero_is_allowed_where_it_means_disabled_or_no_retry():
    p = load_timeout_policies(
        {},
        overrides={
            "sse": {"heartbeat_seconds": 0},
            "llm": {"local_generation_timeout_seconds": 0, "grounded_max_retries": 0},
            "tool_loop": {"recovery": {"max_retries": 0}},
        },
    )
    assert p.sse.heartbeat_seconds == 0.0
    assert p.llm.local_generation_timeout_seconds == 0.0
    assert p.llm.grounded_max_retries == 0
    assert p.tool_loop.recovery.max_retries == 0


@pytest.mark.parametrize(
    "env, attr, value",
    [
        (
            {"TOOL_APPROVAL_TIMEOUT_SECONDS": "12.5"},
            ("tool_loop", "approval_timeout_seconds"),
            12.5,
        ),
        (
            {"AGENTIC_SEARCH_GENERATION_TIMEOUT": "0"},
            ("llm", "local_generation_timeout_seconds"),
            0.0,
        ),
        (
            {"LLM_SOCKET_READ_TIMEOUT": "45"},
            ("llm", "socket_read_timeout_seconds"),
            45.0,
        ),
        (
            {"AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS": "7"},
            ("sse", "heartbeat_seconds"),
            7.0,
        ),
    ],
)
def test_env_var_beats_the_file(tmp_path, env, attr, value):
    f = tmp_path / "t.toml"
    f.write_text(
        "[llm]\nsocket_read_timeout_seconds = 1\nlocal_generation_timeout_seconds = 1\n"
        "[tool_loop]\napproval_timeout_seconds = 1\n[sse]\nheartbeat_seconds = 1\n"
    )
    p = load_timeout_policies({TIMEOUTS_PATH_ENV: str(f), **env})
    assert getattr(getattr(p, attr[0]), attr[1]) == value


@pytest.mark.parametrize("raw", ["0", "-1", "inf", "nan", "abc"])
def test_bad_approval_env_names_the_env_var(raw):
    with pytest.raises(ValueError, match="TOOL_APPROVAL_TIMEOUT_SECONDS"):
        load_timeout_policies({"TOOL_APPROVAL_TIMEOUT_SECONDS": raw})


def test_cache_and_use_and_reset(monkeypatch):
    monkeypatch.setenv("LLM_SOCKET_READ_TIMEOUT", "33")
    reset_timeout_policies()
    assert get_timeout_policies().llm.socket_read_timeout_seconds == 33.0
    custom = load_timeout_policies({}, overrides={"rerank": {"timeout_seconds": 2}})
    with use_timeout_policies(custom):
        assert get_timeout_policies().rerank.timeout_seconds == 2.0
    assert get_timeout_policies().llm.socket_read_timeout_seconds == 33.0


def test_tomli_fallback_loads_on_python_310_path():
    import subprocess
    import sys

    code = (
        "import sys, importlib.abc\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'tomllib': raise ImportError('blocked')\n"
        "sys.meta_path.insert(0, B())\n"
        "from src.internal.configs.timeouts import load_timeout_policies, tomllib\n"
        "assert tomllib.__name__ == 'tomli', tomllib.__name__\n"
        "print(load_timeout_policies({}).rerank.timeout_seconds)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "10.0"


def test_app_settings_carry_the_same_policies():
    from src.internal.configs import load_app_settings

    s = load_app_settings({"TOOL_APPROVAL_TIMEOUT_SECONDS": "12.5"})
    assert s.timeouts.tool_loop.approval_timeout_seconds == 12.5
    assert s.tool_approval_timeout_seconds == 12.5
    assert (
        s.generation_timeout_seconds
        == s.timeouts.llm.local_generation_timeout_seconds
        == 120.0
    )
