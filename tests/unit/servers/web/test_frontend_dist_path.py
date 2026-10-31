"""The backend serves a built frontend bundle only when told where it is.

The frontend lives in its own repository (Agentic-Search-UI) and normally ships
as a separate nginx image. `AGENTIC_SEARCH_FRONTEND_DIST` lets this process
serve a `dist/` directly; unset or pointing at a broken build, the caller falls
back to the inline shell.

These tests fabricate the directory rather than relying on a real build.
"""

from __future__ import annotations

from pathlib import Path

from src.internal.servers.web import app as web_app


def _fake_bundle(root: Path, *, with_assets: bool = True) -> Path:
    dist = root / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html>")
    if with_assets:
        (dist / "assets").mkdir()
    return dist


def test_the_configured_bundle_is_served(tmp_path, monkeypatch):
    dist = _fake_bundle(tmp_path)
    monkeypatch.setenv("AGENTIC_SEARCH_FRONTEND_DIST", str(dist))

    assert web_app._frontend_dist_path() == dist.resolve()


def test_unset_returns_none(monkeypatch):
    """The control: with nothing configured, the caller serves the shell."""
    monkeypatch.delenv("AGENTIC_SEARCH_FRONTEND_DIST", raising=False)

    assert web_app._frontend_dist_path() is None


def test_an_incomplete_bundle_is_not_served(tmp_path, monkeypatch):
    """index.html without assets/ is a broken build, not a servable bundle."""
    dist = _fake_bundle(tmp_path, with_assets=False)
    monkeypatch.setenv("AGENTIC_SEARCH_FRONTEND_DIST", str(dist))

    assert web_app._frontend_dist_path() is None
