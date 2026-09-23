"""The serving baseline must stay the baseline.

`requirements.txt` is the only file the container installs, so anything added to
it lands in the image. These tests pin the split rather than the contents: they
check that the heavy backends stay out of the baseline, and that every package
the baseline drops is either in a companion file or genuinely unimported.

Guarding this matters because the file already carried "Server/runtime" and
"Training / data / retrieval" section comments and nothing enforced them -- the
image shipped faiss-cpu, pyserini, datasets, mlx-lm and playwright for months.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "requirements.txt"
HEAVY = REPO / "requirements-retrieval-heavy.txt"
TRAINING = REPO / "requirements-training.txt"
UNIT_TEST = REPO / "requirements-unit-test.txt"


def _packages(path: Path) -> set[str]:
    names = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!\[;]", line, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def test_the_heavy_backends_stay_out_of_the_serving_baseline():
    """faiss-cpu and pyserini are ~GBs the image cannot use.

    Both are reachable from servers/retrieval/server.py, so they are real
    dependencies of a backend -- just not of the demo/hybrid servers the compose
    stack runs, and pyserini needs a JVM the image does not install.
    """
    baseline = _packages(BASELINE)
    for package in ("faiss-cpu", "pyserini"):
        assert package not in baseline, (
            f"{package} is back in the serving baseline, so it is back in the "
            f"image; it belongs in requirements-retrieval-heavy.txt"
        )
        assert package in _packages(HEAVY)


def test_training_only_dependencies_stay_out_of_the_serving_baseline():
    baseline = _packages(BASELINE)
    for package in ("datasets", "pyarrow"):
        assert package not in baseline, f"{package} belongs in the training file"
        assert package in _packages(TRAINING)


def test_the_dropped_packages_stay_dropped():
    """mlx-lm is imported nowhere; playwright was the wrong package entirely.

    web_search/browser.py shells out to a `playwright-cli` binary through
    subprocess -- the pip wheel named `playwright` does not provide it, so
    declaring it satisfied nothing.
    """
    everywhere = _packages(BASELINE) | _packages(HEAVY) | _packages(TRAINING)
    for package in ("mlx-lm", "playwright"):
        assert package not in everywhere, (
            f"{package} came back; it is imported nowhere in src/, examples/ or "
            f"tests/. If something now needs it, say which in the requirement."
        )


def test_the_baseline_keeps_what_is_required_without_being_imported():
    """Two packages look droppable to an import scan and are not.

    FastAPI needs python-multipart for the UploadFile routes in
    servers/enterprise_settings/api.py, and pytest-asyncio is load-bearing
    through asyncio_mode = "auto" in pyproject.toml.
    """
    baseline = _packages(BASELINE)
    for package in ("python-multipart", "pytest-asyncio", "urllib3"):
        assert package in baseline, (
            f"{package} is required despite never being imported"
        )


def test_the_serving_baseline_still_carries_what_the_servers_import():
    baseline = _packages(BASELINE)
    for package in (
        "fastapi",
        "uvicorn",
        "scikit-learn",  # demo.py TF-IDF
        "sentence-transformers",  # hybrid.py dense leg, reranker, memory encoder
        "torch",
        "transformers",
        "redis",
        "httpx",
    ):
        assert package in baseline, f"{package} is imported on the serving path"


def test_the_unit_test_install_is_unaffected_by_the_split():
    """CI installs requirements-unit-test.txt, which never carried the heavy set.

    That is the standing evidence the serving path does not need them: the full
    suite passes without faiss-cpu, pyserini, datasets, mlx-lm or playwright.
    """
    unit_test = _packages(UNIT_TEST)
    for package in ("faiss-cpu", "pyserini", "datasets", "mlx-lm", "playwright"):
        assert package not in unit_test
