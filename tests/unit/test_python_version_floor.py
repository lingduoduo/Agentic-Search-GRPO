"""Don't use names the minimum supported Python does not have.

``pyproject.toml`` declares ``requires-python = ">=3.10"``, but every CI
workflow pins 3.12. So a 3.11-only name imports cleanly for everyone who runs
CI and breaks at import time for anyone on the declared floor -- which is how
``from datetime import UTC`` (3.11+) reached main and took out collection of
187 test modules on a 3.10 interpreter.

This enforces the floor the package advertises. Raise ``requires-python`` and
the check relaxes on its own; the point is that the two agree.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOTS = (Path("src"), Path("tests"), Path("examples"))

# name -> the version that introduced it.
VERSION_GATED = {
    "UTC": (3, 11),  # datetime.UTC; use timezone.utc
    "TaskGroup": (3, 11),  # asyncio.TaskGroup
    "ExceptionGroup": (3, 11),
    "batched": (3, 12),  # itertools.batched
}

# StrEnum is 3.11+ too, but src/context/enums.py already guards it behind a
# try/except ImportError with a fallback, which is the sanctioned pattern.
GUARDED = {Path("src/context/enums.py")}


def _python_floor() -> tuple[int, int]:
    requires = tomllib.loads(Path("pyproject.toml").read_text())["project"][
        "requires-python"
    ]
    match = re.search(r">=\s*(\d+)\.(\d+)", requires)
    assert match, f"cannot parse requires-python: {requires!r}"
    return int(match.group(1)), int(match.group(2))


def _imported_names() -> list[tuple[str, int, str]]:
    """Every `from <module> import <name>` under the source roots."""
    found = []
    for root in ROOTS:
        for path in root.rglob("*.py"):
            if path in GUARDED:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        found.append((str(path), node.lineno, alias.name))
    return found


def test_requires_python_still_declares_the_floor_this_guards():
    """Guard the guard: if the floor moves to 3.11 this check stops mattering."""
    assert _python_floor() == (3, 10)


def test_no_module_imports_a_name_newer_than_the_supported_floor():
    floor = _python_floor()
    offenders = []
    for path, lineno, name in _imported_names():
        introduced = VERSION_GATED.get(name)
        if introduced and introduced > floor:
            version = ".".join(str(part) for part in introduced)
            offenders.append(f"{path}:{lineno} imports {name} (Python {version}+)")

    floor_text = ".".join(str(part) for part in floor)
    assert not offenders, (
        f"these need Python newer than the declared floor {floor_text}: "
        + "; ".join(sorted(offenders))
    )
