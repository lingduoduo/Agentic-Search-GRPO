"""Report which endpoints the integration suite calls that the app no longer serves.

The integration tree predates several subsystem removals -- CLAUDE.md records the
async worker fleet and the servers/indexing pipeline going -- and the tests were
never in the default pytest run (`testpaths` covers tests/unit and
tests/regression), so nothing has flagged the drift. #630 retired the indexing
directory after establishing that no stack could make it pass.

Sizing the rest needs the endpoints, and an earlier regex pass got this wrong:
matching `/[A-Za-z0-9_{}/-]+` against `f"{API_SERVER_URL}/admin/api-key/{api_key}"`
truncates at the brace and yields `/admin/api-key/{api_key`, which matches no
route for the wrong reason. This walks the AST instead: every f-string whose
parts reference API_SERVER_URL is reconstructed with `{}` for each interpolation,
and app routes are normalised the same way before comparison.

    python -m examples.audit_integration_endpoints
    python -m examples.audit_integration_endpoints --show-files
    python -m examples.audit_integration_endpoints --format json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INTEGRATION = REPO / "tests" / "integration"
_SERVER_URL_NAMES = {"API_SERVER_URL", "MCP_SERVER_URL"}
_PATH_RE = re.compile(r"^/[A-Za-z0-9_./{}-]*$")


def _normalise(path: str) -> str:
    """Compare paths only: drop any query string, collapse parameter names.

    Without the query strip, `/admin/chat-session-history?{}` and
    `/?transportType=streamable-http` are reported as removed when both paths are
    served -- a false positive of exactly the kind this script exists to avoid.
    """
    path = path.split("?", 1)[0]
    return re.sub(r"\{[^}]*\}", "{}", path.rstrip("/")) or "/"


def _references_server_url(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in _SERVER_URL_NAMES:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in _SERVER_URL_NAMES:
            return True
    return False


def _template_from_joinedstr(node: ast.JoinedStr) -> str:
    """Rebuild an f-string as text, with `{}` standing in for each interpolation."""
    out = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            out.append(value.value)
        else:
            out.append("{}")
    return "".join(out)


def _url_prefix_constants(tree: ast.AST) -> dict[str, str]:
    """Module-level names assigned a server URL plus a path prefix.

    Managers do this rather than inlining the full path:

        DISCORD_BOT_API_URL = f"{API_SERVER_URL}/manage/admin/discord-bot"
        ... f"{DISCORD_BOT_API_URL}/config"

    Without following the constant, the second f-string looks like it builds
    `/config` -- or, if the prefix is the whole value, like it builds nothing at
    all. That is what left 19 files with no attributable endpoint.
    """
    prefixes: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.JoinedStr) and _references_server_url(value):
            text = _template_from_joinedstr(value)
            tail = text[text.index("{}") + 2 :] if "{}" in text else ""
            prefixes[target.id] = tail
        elif (
            isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Add)
            and _references_server_url(value.left)
            and isinstance(value.right, ast.Constant)
            and isinstance(value.right.value, str)
        ):
            prefixes[target.id] = value.right.value
    return prefixes


def _paths_in(tree: ast.AST, prefixes: dict[str, str] | None = None) -> set[str]:
    """Endpoint paths this module builds against a server URL or a derived name."""
    found: set[str] = set()
    prefixes = prefixes or {}

    for node in ast.walk(tree):
        # f"{API_SERVER_URL}/manage/admin/cc-pair/{id}/status"
        if isinstance(node, ast.JoinedStr):
            base = ""
            if _references_server_url(node):
                pass  # the server URL is the first interpolation
            else:
                named = next(
                    (
                        prefixes[sub.id]
                        for sub in ast.walk(node)
                        if isinstance(sub, ast.Name) and sub.id in prefixes
                    ),
                    None,
                )
                if named is None:
                    continue
                base = named
            text = _template_from_joinedstr(node)
            tail = text[text.index("{}") + 2 :] if "{}" in text else text
            endpoint = base + tail
            if endpoint.startswith("/"):
                found.add(endpoint)
        # API_SERVER_URL + "/manage/admin/deletion-attempt"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if _references_server_url(node.left) and isinstance(
                node.right, ast.Constant
            ):
                if isinstance(node.right.value, str) and node.right.value.startswith(
                    "/"
                ):
                    found.add(node.right.value)
    return found


def endpoints_by_callable(tree: ast.AST) -> dict[str, set[str]]:
    """Endpoints built inside each callable, keyed "Class.method" or "function".

    Managers are classes whose methods each hit specific endpoints --
    ``CCPairManager.pause_cc_pair`` builds ``/manage/admin/cc-pair/{}/status``
    and nothing else. Resolving to the method rather than the module is what
    turns "this test imports a manager that touches dead endpoints" into "this
    test calls a method that does".
    """
    out: dict[str, set[str]] = defaultdict(set)

    prefixes = _url_prefix_constants(tree)

    def _record(qualname: str, node: ast.AST) -> None:
        for endpoint in _paths_in(node, prefixes):
            out[qualname].add(endpoint)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _record(f"{node.name}.{item.name}", item)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # module-level helper; skip methods, already handled above
            if not any(
                item is node
                for cls in ast.walk(tree)
                if isinstance(cls, ast.ClassDef)
                for item in cls.body
            ):
                _record(node.name, node)
    return out


def callables_invoked(tree: ast.AST) -> set[str]:
    """Names this module calls, as "Class.method" and bare "function"."""
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            called.add(f"{func.value.id}.{func.attr}")
            called.add(func.attr)
        elif isinstance(func, ast.Name):
            called.add(func.id)
    return called


def _imported_helper_modules(tree: ast.AST) -> set[str]:
    """common_utils modules this file imports, as dotted suffixes.

    Most tests never build a URL themselves -- they call CCPairManager and
    friends. Without this, two thirds of the suite is unclassifiable and the
    "every endpoint gone" count reads as a total when it is a floor.
    """
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "common_utils" in node.module:
                mods.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "common_utils" in alias.name:
                    mods.add(alias.name)
    return mods


def collect() -> dict[str, set[str]]:
    """endpoint template -> the integration files that build it."""
    by_endpoint: dict[str, set[str]] = defaultdict(set)
    for path in sorted(INTEGRATION.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        rel = path.relative_to(REPO).as_posix()
        for endpoint in _paths_in(tree, _url_prefix_constants(tree)):
            if _PATH_RE.match(endpoint.split("?", 1)[0]):
                by_endpoint[endpoint].add(rel)
    return by_endpoint


def app_routes() -> set[str]:
    from src.internal.servers.web.app import create_web_app

    return {
        _normalise(route.path)
        for route in create_web_app().routes
        if hasattr(route, "path")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-files", action="store_true")
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help=(
            "Attribute only endpoints a file builds itself. This is the floor: "
            "most tests call a manager method rather than constructing a URL, so "
            "it under-counts. The default resolves those calls to the method and "
            "is the precise view."
        ),
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    by_endpoint = collect()
    served = app_routes()

    # endpoints each helper CALLABLE builds -- "CCPairManager.delete" etc.
    helper_callables: dict[str, set[str]] = defaultdict(set)
    for path in sorted((INTEGRATION / "common_utils").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for qualname, endpoints in endpoints_by_callable(tree).items():
            helper_callables[qualname] |= endpoints
    gone = {e: f for e, f in by_endpoint.items() if _normalise(e) not in served}
    alive = {e for e in by_endpoint if _normalise(e) in served}

    test_files = {
        p.relative_to(REPO).as_posix() for p in INTEGRATION.rglob("test_*.py")
    }
    # The decision-relevant split: a file whose every endpoint is gone cannot be
    # repaired, only retired. A file with a mix needs judgement per test.
    per_file: dict[str, dict[str, set[str]]] = {
        f: {"served": set(), "gone": set()} for f in test_files
    }

    def _bucket(endpoint: str) -> str:
        return "served" if _normalise(endpoint) in served else "gone"

    for endpoint, files in by_endpoint.items():
        for f in files & test_files:
            per_file[f][_bucket(endpoint)].add(endpoint)

    # Fold in the endpoints each test reaches through the helper methods it calls.
    for f in test_files if not args.direct_only else []:
        try:
            tree = ast.parse((REPO / f).read_text())
        except SyntaxError:
            continue
        for name in callables_invoked(tree):
            for endpoint in helper_callables.get(name, ()):
                per_file[f][_bucket(endpoint)].add(endpoint)

    wholly_dead = sorted(
        f for f, b in per_file.items() if b["gone"] and not b["served"]
    )
    mixed = sorted(f for f, b in per_file.items() if b["gone"] and b["served"])
    clean = sorted(f for f, b in per_file.items() if not b["gone"] and b["served"])
    no_endpoints = sorted(
        f for f, b in per_file.items() if not b["gone"] and not b["served"]
    )
    files_with_dead = set(wholly_dead) | set(mixed)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "endpoints_referenced": len(by_endpoint),
                    "served": sorted(alive),
                    "gone": {e: sorted(f) for e, f in sorted(gone.items())},
                    "test_files_total": len(test_files),
                    "test_files_touching_removed": sorted(files_with_dead),
                    "wholly_removed_api": wholly_dead,
                    "mixed": mixed,
                    "fully_served": clean,
                    "no_endpoints_referenced": no_endpoints,
                },
                indent=2,
            )
        )
        return

    mode = (
        "direct references only"
        if args.direct_only
        else "resolved through called helper methods"
    )
    print(f"attribution          : {mode}")
    print(f"endpoints referenced : {len(by_endpoint)}")
    print(f"  served by the app  : {len(alive)}")
    print(f"  no longer served   : {len(gone)}")
    print(f"test files           : {len(test_files)}")
    print(f"  every endpoint gone: {len(wholly_dead)}   (retire; cannot be repaired)")
    print(f"  mixed              : {len(mixed)}   (judgement per test)")
    print(f"  fully served       : {len(clean)}")
    print(f"  no endpoints found : {len(no_endpoints)}   (helpers, or indirect calls)")
    if gone:
        print("\nno longer served:")
        for endpoint in sorted(gone):
            print(f"  {endpoint}")
            if args.show_files:
                for f in sorted(gone[endpoint]):
                    print(f"      {f}")


if __name__ == "__main__":
    main()
