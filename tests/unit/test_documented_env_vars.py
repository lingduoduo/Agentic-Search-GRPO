"""Every env var documented in configuration.md must be read somewhere.

Six documented variables did nothing when set, two of them inside
copy-pasteable startup commands. That fails the way a correct setting looks
when it happens not to matter -- no error, no warning -- so the operator
concludes the knob does not help rather than that they never turned it.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG_DOC = REPO / "docs" / "configuration.md"

# A documented variable read nowhere in the tree is a bug unless there is a
# reason. The value is that reason -- an exemption must be argued, not just
# added.
ALLOWED_UNREFERENCED: dict[str, str] = {}

_SEARCH_ROOTS = ("src", "examples", "tests", ".github")
_SEARCH_SUFFIXES = {".py", ".ts", ".tsx", ".sh", ".yml", ".yaml", ".go", ".toml"}

_DOC_ROW = re.compile(r"^\|\s*`([A-Z][A-Z0-9_]{3,})`\s*\|", re.M)
_TOKEN = re.compile(r"\b([A-Z][A-Z0-9_]{3,})\b")


def _documented_env_vars() -> set[str]:
    return set(_DOC_ROW.findall(CONFIG_DOC.read_text(encoding="utf-8")))


def _referenced_env_vars() -> set[str]:
    seen: set[str] = set()
    for root in _SEARCH_ROOTS:
        for path in (REPO / root).rglob("*"):
            if not path.is_file() or path.suffix not in _SEARCH_SUFFIXES:
                continue
            if "node_modules" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            seen.update(_TOKEN.findall(text))
    return seen


def test_every_documented_env_var_is_read_somewhere():
    """A setting that nothing reads is not a setting."""
    documented = _documented_env_vars()
    assert len(documented) > 50, (
        "the table parser found almost nothing; check the regex"
    )

    unreferenced = sorted(
        name
        for name in documented - _referenced_env_vars()
        if name not in ALLOWED_UNREFERENCED
    )

    assert not unreferenced, (
        "documented in configuration.md but read nowhere in "
        f"{', '.join(r + '/' for r in _SEARCH_ROOTS)}: {unreferenced}. "
        "Either fix the name, move it to where its real flag is documented, "
        "delete the row, or add it to ALLOWED_UNREFERENCED with a reason."
    )


def test_every_allowlist_entry_carries_a_reason():
    """An exemption must be argued, not just added."""
    for name, reason in ALLOWED_UNREFERENCED.items():
        assert reason.strip(), f"{name} is exempted without a reason"


def test_the_allowlist_has_no_stale_entries():
    """An entry that is now referenced, or no longer documented, must go."""
    documented = _documented_env_vars()
    referenced = _referenced_env_vars()
    for name in ALLOWED_UNREFERENCED:
        assert name in documented, f"{name} is exempted but no longer documented"
        assert name not in referenced, f"{name} is exempted but is now read; drop it"
