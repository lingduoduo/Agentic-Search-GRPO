from __future__ import annotations

from pathlib import Path

import pytest

from src.internal.configs import AppSettings
from src.internal.configs import Tier
from src.internal.configs import get_env_bool
from src.internal.configs import is_license_enforcement_exempt
from src.internal.configs import is_path_allowed_for_gated_tenant
from src.internal.configs import is_path_allowed_for_tier
from src.internal.configs import load_app_settings
from src.internal.configs import load_permission_sync_settings
from src.internal.configs import required_tier_for_path
from src.internal.configs.default_config import DEFAULT_CONFIG
from src.internal.servers.web.app import SearchExperienceSettings


def test_load_app_settings_reads_typed_environment():
    settings = load_app_settings(
        {
            "AGENTIC_SEARCH_RETRIEVAL_URL": "http://search.test/retrieve",
            "AGENTIC_SEARCH_WEB_TOP_K": "7",
            "AGENTIC_SEARCH_AUTH_SECRET": "secret",
            "AGENTIC_SEARCH_SUPER_USERS": '["admin@example.test"]',
            "POSTHOG_DEBUG_LOGS_ENABLED": "true",
        }
    )

    assert settings.services.retrieval_url == "http://search.test/retrieve"
    assert settings.services.web_top_k == 7
    assert settings.auth.secret == "secret"
    assert settings.auth.super_users == ("admin@example.test",)
    assert settings.telemetry.posthog_debug_logs_enabled is True


def test_load_app_settings_reads_intent_index_configuration():
    settings = load_app_settings(
        {
            "AGENTIC_SEARCH_INTENT_INDEX_PATH": "/models/intent-index",
        }
    )

    assert settings.intent_index_path == Path("/models/intent-index")


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
@pytest.mark.parametrize(
    "key,field",
    [
        ("AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN", "intent_min_route_margin"),
        ("AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE", "intent_min_module_score"),
    ],
)
def test_intent_threshold_must_be_finite_probability(key: str, field: str, value: str):
    """Retargeted onto the surviving thresholds.

    These asserted `intent_model_min_confidence` until that gate was removed for
    changing 3 decisions in 416. The validation itself is worth keeping — a NaN
    threshold silently disables a gate rather than failing — so it now covers the
    two thresholds that still exist.
    """
    with pytest.raises(ValueError, match=field):
        load_app_settings({key: value})


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "field", ["intent_min_route_margin", "intent_min_module_score"]
)
def test_explicit_intent_threshold_must_be_finite_probability(field: str, value: float):
    with pytest.raises(ValueError, match=field):
        AppSettings(**{field: value})


def test_intent_top_k_defaults_to_the_selected_value_and_reads_from_env():
    """15 since #522, when k was first chosen on the tuning/test split.

    It was 3 from #511 to #521 -- an arbitrary constant nobody had swept.
    """
    assert load_app_settings({}).intent_top_k == 8
    assert load_app_settings({"AGENTIC_SEARCH_INTENT_TOP_K": "8"}).intent_top_k == 8


@pytest.mark.parametrize("value", ["0", "-1"])
def test_intent_top_k_must_be_a_positive_integer(value: str):
    with pytest.raises(ValueError, match="intent_top_k"):
        load_app_settings({"AGENTIC_SEARCH_INTENT_TOP_K": value})


def test_explicit_intent_top_k_must_be_a_positive_integer():
    with pytest.raises(ValueError, match="intent_top_k"):
        AppSettings(intent_top_k=0)


def test_default_config_documents_intent_model_configuration():
    """The mirror must carry every routing threshold, at the tuned values."""
    settings = load_app_settings({})
    assert DEFAULT_CONFIG["AGENTIC_SEARCH_INTENT_INDEX_PATH"] == ""
    assert (
        DEFAULT_CONFIG["AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN"]
        == settings.intent_min_route_margin
        == 0.010
    )
    assert (
        DEFAULT_CONFIG["AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE"]
        == settings.intent_min_module_score
        == 0.8215
    )
    assert DEFAULT_CONFIG["AGENTIC_SEARCH_INTENT_TOP_K"] == settings.intent_top_k == 8


def test_route_clarification_defaults_to_true_and_reads_from_env():
    assert load_app_settings({}).route_clarification is True
    assert (
        load_app_settings(
            {"AGENTIC_SEARCH_ROUTE_CLARIFICATION": "false"}
        ).route_clarification
        is False
    )
    assert DEFAULT_CONFIG["AGENTIC_SEARCH_ROUTE_CLARIFICATION"] is True


def test_bool_env_parser_rejects_ambiguous_values():
    assert get_env_bool({"FLAG": "yes"}, "FLAG") is True
    assert get_env_bool({"FLAG": "0"}, "FLAG", True) is False

    with pytest.raises(ValueError, match="FLAG"):
        get_env_bool({"FLAG": "sometimes"}, "FLAG")


def test_permission_sync_settings_keep_connector_specific_defaults():
    settings = load_permission_sync_settings(
        {
            "DEFAULT_PERMISSION_DOC_SYNC_FREQUENCY": "42",
            "CONFLUENCE_PERMISSION_DOC_SYNC_FREQUENCY": "1800",
            "GITHUB_PERMISSION_GROUP_SYNC_FREQUENCY": "60",
            "CONFLUENCE_ANONYMOUS_ACCESS_IS_PUBLIC": "true",
        }
    )

    assert settings.doc_sync_frequency("unknown") == 42
    assert settings.doc_sync_frequency("confluence") == 1800
    assert settings.group_sync_frequency("github") == 60
    assert settings.anonymous_access_is_public("confluence") is True


def test_license_tier_gating_uses_longest_prefix_and_allowlist():
    assert is_license_enforcement_exempt("/health/live")
    assert required_tier_for_path("/admin/hooks/outbound") == Tier.ENTERPRISE
    assert is_path_allowed_for_tier("/admin/api-key", Tier.BUSINESS)
    assert not is_path_allowed_for_tier("/analytics/team", Tier.BUSINESS)
    assert is_path_allowed_for_tier("/analytics/team", Tier.ENTERPRISE)


def test_multi_tenant_gating_allowlist_is_prefix_based():
    assert is_path_allowed_for_gated_tenant("/tenants/create-subscription-session")
    assert is_path_allowed_for_gated_tenant("/assets/app.css")
    assert not is_path_allowed_for_gated_tenant("/api/agent")


def test_load_app_settings_reads_search_agent_config():
    settings = load_app_settings(
        {
            "SEARCH_AGENT_MODEL": "Qwen/Qwen2.5-1.5B-Instruct",
            "SEARCH_AGENT_DEVICE": "mps",
        }
    )
    assert settings.search_agent_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert settings.search_agent_device == "mps"


def test_load_app_settings_search_agent_defaults_to_none_and_mps():
    settings = load_app_settings({})
    assert settings.search_agent_model is None
    assert settings.search_agent_device == "mps"
    assert settings.search_agent_server_url is None


def test_load_app_settings_reads_search_agent_server_url():
    settings = load_app_settings(
        {
            "SEARCH_AGENT_SERVER_URL": "https://xxxx.ngrok-free.app",
            "SEARCH_AGENT_MODEL": "Qwen/Qwen2.5-1.5B-Instruct",
        }
    )
    assert settings.search_agent_server_url == "https://xxxx.ngrok-free.app"
    assert settings.search_agent_model == "Qwen/Qwen2.5-1.5B-Instruct"


def test_load_app_settings_reads_rerank_url():
    assert (
        load_app_settings(
            {"AGENTIC_SEARCH_RERANK_URL": "http://localhost:8002/rerank"}
        ).services.rerank_url
        == "http://localhost:8002/rerank"
    )
    # Unset ⇒ None (rerank stays off by default).
    assert load_app_settings({}).services.rerank_url is None


def test_load_app_settings_reads_browser_search_url():
    assert (
        load_app_settings(
            {"AGENTIC_SEARCH_BROWSER_SEARCH_URL": "http://localhost:8003/retrieve"}
        ).services.browser_search_url
        == "http://localhost:8003/retrieve"
    )
    # Unset ⇒ None (the /api/agent browser leg stays off by default).
    assert load_app_settings({}).services.browser_search_url is None


def test_web_settings_can_be_built_from_app_settings():
    app_settings = load_app_settings(
        {
            "AGENTIC_SEARCH_RETRIEVAL_URL": "http://search.test/retrieve",
            "AGENTIC_SEARCH_WEB_TOP_K": "9",
            "AGENTIC_SEARCH_WEB_DB_PATH": "/tmp/search.sqlite3",
            "AGENTIC_SEARCH_RERANK_URL": "http://rr.test/rerank",
            "AGENTIC_SEARCH_BROWSER_SEARCH_URL": "http://browser.test/retrieve",
        }
    )

    web_settings = SearchExperienceSettings.from_app_settings(app_settings)

    assert web_settings.search_url == "http://search.test/retrieve"
    assert web_settings.top_k == 9
    assert web_settings.db_path == "/tmp/search.sqlite3"
    assert web_settings.rerank_url == "http://rr.test/rerank"
    assert web_settings.browser_search_url == "http://browser.test/retrieve"


def test_tool_agent_parser_defaults_to_json():
    settings = load_app_settings({})
    assert settings.tool_agent_parser == "json"


def test_tool_agent_parser_reads_from_env():
    settings = load_app_settings({"TOOL_AGENT_PARSER": "hermes"})
    assert settings.tool_agent_parser == "hermes"


def test_tool_approval_timeout_defaults_to_sixty_seconds() -> None:
    assert load_app_settings({}).tool_approval_timeout_seconds == 60.0


def test_tool_approval_timeout_reads_environment() -> None:
    settings = load_app_settings({"TOOL_APPROVAL_TIMEOUT_SECONDS": "12.5"})
    assert settings.tool_approval_timeout_seconds == 12.5


def test_tool_approval_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="TOOL_APPROVAL_TIMEOUT_SECONDS"):
        load_app_settings({"TOOL_APPROVAL_TIMEOUT_SECONDS": "0"})


@pytest.mark.parametrize("value", ["nan", "inf", "+inf", "-inf"])
def test_tool_approval_timeout_must_be_finite(value: str) -> None:
    with pytest.raises(ValueError, match="TOOL_APPROVAL_TIMEOUT_SECONDS"):
        load_app_settings({"TOOL_APPROVAL_TIMEOUT_SECONDS": value})
