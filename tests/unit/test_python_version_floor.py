"""Don't use names the minimum supported Python does not have.

The dependency baseline requires Python 3.12 (NumPy 2.5+). Keep source imports
compatible with the declared floor as newer interpreters add standard names.

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
    """Keep the declared floor aligned with the NumPy dependency baseline."""
    assert _python_floor() == (3, 12)


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


def _ci_unit_test_matrix() -> list[tuple[int, int]]:
    """The Python versions the unit-test job runs on.

    Parsed with a regex rather than PyYAML: pyyaml is importable here but is
    not a declared dependency, and reaching for an undeclared import is the
    mistake #607 fixed.
    """
    workflow = Path(".github/workflows/ci.yml").read_text()
    block = re.search(
        r"unit-tests:.*?python-version:\s*\[([^\]]+)\]", workflow, re.DOTALL
    )
    assert block, "cannot find the unit-test job's python-version matrix"
    return [
        (int(major), int(minor))
        for major, minor in re.findall(r'"(\d+)\.(\d+)"', block.group(1))
    ]


def test_ci_runs_the_floor_it_claims_to_support():
    """Testing only the newest version is how 3.11-only code reached main.

    A static guard catches names that do not exist. It cannot catch behaviour
    that differs -- `asyncio.TimeoutError` merging into the builtin in 3.11, or
    `sum()` gaining compensation in 3.12 -- and only running there does.
    """
    floor = _python_floor()
    matrix = _ci_unit_test_matrix()

    assert floor in matrix, (
        f"requires-python declares {floor[0]}.{floor[1]} but CI runs "
        f"{['.'.join(map(str, v)) for v in matrix]}"
    )
