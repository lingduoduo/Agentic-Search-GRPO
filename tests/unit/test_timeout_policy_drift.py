"""Schema-owned timeout/retry values must not reappear as literals."""

import ast
import pathlib

import pytest

MIGRATED = [
    "src/internal/tools/public_data/_http.py",
    "src/internal/tools/public_data/geo.py",
    "src/internal/tools/search.py",
    "src/internal/tools/routing_tools.py",
    "src/internal/tools/api.py",
    "src/internal/tools/mcp_client.py",
    "src/context/retrieval/client.py",
    "src/context/retrieval/search_runner.py",
    "src/agents/search/search.py",
    "src/agents/generation/single_turn.py",
    "src/internal/search/stages.py",
    "src/internal/llm/multi_llm.py",
    "src/model/serving.py",
    "src/agents/search/agentic_rag.py",
    "src/context/models.py",
    "src/agents/tool/tool_calling.py",
    "src/agents/tool/recovery.py",
    "src/internal/servers/web/tool_approval.py",
    "src/context/tool_evidence.py",
    "src/context/pipeline.py",
    "src/internal/servers/sse.py",
    "src/internal/servers/web/app.py",
    "src/internal/servers/web/tool_agent_runner.py",
]
POLICY_WORDS = (
    "timeout",
    "retries",
    "attempts",
    "backoff",
    "retry_budget",
    "heartbeat",
)

# Non-schema hits the heuristic flags in MIGRATED files. Each is a literal that
# is NOT one of the values in timeouts.toml / spec §2 — never widen silently.
ALLOWED = {
    # PublicDataError.attempts: how many attempts a raised error reports having
    # made (a bookkeeping default of 1), not the tools.public_data.max_attempts
    # retry-loop policy.
    "src/internal/tools/public_data/_http.py:67 attempts",
    # _unavailable_result's `retries` parameter: the retry_count to report on an
    # already-decided unavailable outcome (0 = none happened), not a policy default.
    "src/agents/tool/tool_calling.py:540 retries",
    # RecoveryState.retries: a per-run counter initialized to 0, not a
    # tool_loop.recovery.max_retries policy default.
    "src/agents/tool/recovery.py:88 retries",
    # heartbeat_thread.join(timeout=0.1): grace period to let the heartbeat
    # thread notice stop_event after generation finishes, not the
    # llm.local_heartbeat_seconds interval itself (already read from policy a
    # few lines above) and not any other schema key.
    "src/model/serving.py:568 timeout",
}


def _is_policy_name(name: str) -> bool:
    lowered = name.lower()
    return any(w in lowered for w in POLICY_WORDS)


def _numeric(node) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ) or (
        isinstance(node, ast.Tuple)
        and node.elts
        and all(_numeric(e) for e in node.elts)
    )


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _violations(path: str) -> list[str]:
    tree = ast.parse(pathlib.Path(path).read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            is_client_timeout = (_call_name(node) or "").endswith("ClientTimeout")
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                if not (
                    _is_policy_name(kw.arg) or (kw.arg == "total" and is_client_timeout)
                ):
                    continue
                if _numeric(kw.value):
                    found.append(f"{path}:{kw.value.lineno} {kw.arg}")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            pos = args.posonlyargs + args.args
            pairs = list(zip(pos[len(pos) - len(args.defaults) :], args.defaults))
            pairs += [
                (a, d)
                for a, d in zip(args.kwonlyargs, args.kw_defaults)
                if d is not None
            ]
            found += [
                f"{path}:{d.lineno} {a.arg}"
                for a, d in pairs
                if _is_policy_name(a.arg) and _numeric(d)
            ]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if (
                _is_policy_name(node.target.id)
                and node.value is not None
                and _numeric(node.value)
            ):
                found.append(f"{path}:{node.lineno} {node.target.id}")
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Name)
                    and t.id.isupper()
                    and _is_policy_name(t.id)
                    and _numeric(node.value)
                ):
                    found.append(f"{path}:{node.lineno} {t.id}")
    return found


@pytest.mark.parametrize("path", MIGRATED)
def test_no_literal_policy_defaults(path):
    violations = [v for v in _violations(path) if v not in ALLOWED]
    assert violations == []


def test_guard_catches_a_literal(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def f(*, timeout_seconds: int = 15):\n    pass\n")
    assert _violations(str(f)) == [f"{f}:1 timeout_seconds"]


def test_guard_catches_a_call_site_literal(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import asyncio\nasyncio.wait_for(x, timeout=8.0)\n")
    assert _violations(str(f)) == [f"{f}:2 timeout"]
