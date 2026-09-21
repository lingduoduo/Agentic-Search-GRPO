"""The HTTP client src/ imports must be the one the dependency files declare.

`httpx` and `httpx2` are unrelated distributions -- `httpx2` is pydantic's
successor, imported as `httpx2`. Declaring one while importing the other leaves
the real dependency satisfied only transitively, through mcp/fastmcp/openai.
"""

import ast
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

REQUIREMENTS_FILES = ("requirements.txt", "requirements-unit-test.txt")


def _top_level_imports(root: Path) -> set[str]:
    """Return every top-level module name imported by the Python files under root."""
    modules: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _declared(requirements: str) -> set[str]:
    """Return the distribution names declared by a requirements file's lines."""
    return {
        match.group(1)
        for match in (
            re.match(r"([A-Za-z0-9._-]+)", line.strip())
            for line in requirements.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        if match
    }


def test_src_imports_httpx_not_httpx2():
    """The premise of the declaration below: src/ uses `httpx`, never `httpx2`."""
    modules = _top_level_imports(Path("src"))

    assert "httpx" in modules
    assert "httpx2" not in modules


def test_requirements_declare_httpx_and_not_httpx2():
    """Both requirements files ship the client src/ actually imports."""
    for name in REQUIREMENTS_FILES:
        declared = _declared(Path(name).read_text())

        assert "httpx" in declared, f"{name} does not declare httpx"
        assert "httpx2" not in declared, f"{name} declares the unimported httpx2"


def test_mcp_extra_declares_httpx():
    """`pip install .[mcp]` must bring the client src/internal/mcp_server imports."""
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    mcp_extra = project["optional-dependencies"]["mcp"]

    assert any(item.startswith("httpx>") for item in mcp_extra)
    assert not any(item.startswith("httpx2") for item in mcp_extra)
