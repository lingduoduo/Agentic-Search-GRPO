"""A browser-search server without playwright-cli must say so, not return nothing.

`browser.py` shells out to a `playwright-cli` binary. Nothing declares it: the
`playwright` pip wheel does not provide it (that ships `playwright`), and the
container installs no such tool -- so the browser provider could never work
there. #628 removed the mis-declared wheel, which made the gap visible without
changing it.

Before this, a missing binary raised FileNotFoundError from subprocess.run,
propagated into `_search_and_process`'s broad `except Exception`, and became
`logger.warning("browser search failed")` with an empty result list. Since
`web_search` cascades serpapi -> browser, that is indistinguishable from "the
query found nothing" -- the single most misleading way for a dependency to be
absent.
"""

from __future__ import annotations

import pytest

from src.internal.servers.web_search.browser import (
    BrowserSearchConfig,
    BrowserSearchEngine,
    BrowserSearchUnavailableError,
    playwright_cli_available,
)


def _engine() -> BrowserSearchEngine:
    return BrowserSearchEngine(BrowserSearchConfig(topk=1, batch_workers=1))


def test_a_missing_binary_raises_an_error_that_names_it(monkeypatch):
    monkeypatch.setattr(
        "src.internal.servers.web_search.browser.PLAYWRIGHT_CMD",
        "playwright-cli-does-not-exist",
    )

    with pytest.raises(BrowserSearchUnavailableError) as excinfo:
        _engine()._run("open", "about:blank")

    message = str(excinfo.value)
    assert "playwright-cli-does-not-exist" in message
    assert "not installed" in message or "not on PATH" in message


def test_the_error_survives_the_search_path_as_a_distinct_signal(monkeypatch):
    """`_search_and_process` degrades every failure to an empty list on purpose.

    That is right for a search target that fails and wrong for a tool that is
    absent, so the absence is re-raised rather than swallowed.
    """
    monkeypatch.setattr(
        "src.internal.servers.web_search.browser.PLAYWRIGHT_CMD",
        "playwright-cli-does-not-exist",
    )

    with pytest.raises(BrowserSearchUnavailableError):
        _engine()._search_and_process("anything")


def test_a_real_search_failure_still_degrades_to_no_results(monkeypatch):
    """The tolerance that swallowing provided must survive for actual failures."""
    engine = _engine()

    def _boom(*args, **kwargs):
        raise RuntimeError("the search target returned garbage")

    monkeypatch.setattr(engine, "_run", _boom)

    assert engine._search_and_process("anything") == []


def test_availability_is_reportable_without_running_a_search(monkeypatch):
    """main() checks this before serving; a server that cannot work should not start."""
    monkeypatch.setattr(
        "src.internal.servers.web_search.browser.PLAYWRIGHT_CMD",
        "playwright-cli-does-not-exist",
    )
    assert playwright_cli_available() is False

    monkeypatch.setattr("src.internal.servers.web_search.browser.PLAYWRIGHT_CMD", "sh")
    assert playwright_cli_available() is True


def test_main_refuses_to_serve_without_the_binary(monkeypatch):
    from src.internal.servers.web_search import browser

    monkeypatch.setattr(browser, "PLAYWRIGHT_CMD", "playwright-cli-does-not-exist")
    monkeypatch.setattr(browser, "load_environment", lambda: None)
    monkeypatch.setattr(
        browser,
        "parse_args",
        lambda: __import__("argparse").Namespace(
            topk=3, workers=2, host="127.0.0.1", port=8003
        ),
    )

    def _must_not_serve(
        *args, **kwargs
    ):  # pragma: no cover - the point is it is not reached
        raise AssertionError("uvicorn started without playwright-cli present")

    monkeypatch.setattr(browser, "run_uvicorn_app", _must_not_serve)

    with pytest.raises(SystemExit) as excinfo:
        browser.main()

    # SystemExit carries the message; the interpreter is what prints it, so the
    # contract to assert is the exception's payload, not captured stderr.
    assert excinfo.value.code != 0
    assert "playwright-cli-does-not-exist" in str(excinfo.value)
