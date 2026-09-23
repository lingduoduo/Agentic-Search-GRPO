"""FastAPI retrieval server that uses playwright-cli for browser-based search."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import quote_plus

from fastapi import FastAPI

from src.internal.servers.app import (
    add_host_port_args,
    create_search_app,
    format_document,
    load_environment,
    run_uvicorn_app,
)

logger = logging.getLogger(__name__)

DEFAULT_TOPK = 5
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
PLAYWRIGHT_CMD = "playwright-cli"


class BrowserSearchUnavailableError(RuntimeError):
    """The playwright-cli binary this server is built around is not present.

    Distinct from a search that failed. `_search_and_process` degrades every
    failure to an empty result list, which is right for a target that returned
    garbage and wrong for a missing tool: `web_search` cascades serpapi ->
    browser, so an empty list reads as "the query found nothing" rather than
    "this provider cannot run".

    Nothing declares the binary. The `playwright` pip wheel does not provide it
    and the container image installs no such tool, so this provider is host-only
    until one of those changes.
    """


def playwright_cli_available() -> bool:
    """Whether PLAYWRIGHT_CMD resolves on PATH, without running a search."""
    return shutil.which(PLAYWRIGHT_CMD) is not None


GOOGLE_SEARCH_URL = "https://www.google.com/search?q={query}&num={topk}&hl=en"
YAHOO_SEARCH_URL = "https://search.yahoo.com/search?p={query}"
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/index.php?search={query}"
SUBPROCESS_TIMEOUT = 30

# Extracts organic results from a Google SERP via h3 headings — more stable than class names.
_GOOGLE_EXTRACT_JS = (
    "JSON.stringify("
    "[...document.querySelectorAll('h3')]"
    ".filter(h=>h.closest('a'))"
    ".slice(0,10)"
    ".map(h=>({title:h.textContent.trim(),"
    "url:h.closest('a').href,"
    "snippet:(h.closest('[data-hveid]')?.lastElementChild?.textContent?.trim()||'')}))"
    ".filter(r=>r.url&&!r.url.includes('google.com/search'))"
    ")"
)
_YAHOO_EXTRACT_JS = (
    "JSON.stringify("
    "[...document.querySelectorAll('#web li, .algo')].slice(0,10)"
    ".map(el=>{const a=el.querySelector('h3 a, a');"
    "return {title:(a?.textContent||'').trim(),url:a?.href||'',"
    "snippet:(el.querySelector('.compText, .fc-falcon, p')?.textContent||'').trim()}})"
    ".filter(r=>r.title&&r.url&&!r.title.toLowerCase().includes('search again'))"
    ")"
)
_WIKIPEDIA_EXTRACT_JS = (
    "JSON.stringify("
    "location.pathname.startsWith('/wiki/')"
    "? [{title:document.title.replace(' - Wikipedia',''),url:location.href,"
    "snippet:[...document.querySelectorAll('p')].map(p=>p.textContent.trim()).find(Boolean)||''}]"
    ": [...document.querySelectorAll('.mw-search-result')].slice(0,10)"
    ".map(el=>{const a=el.querySelector('.mw-search-result-heading a');"
    "return {title:(a?.textContent||'').trim(),url:a?new URL(a.getAttribute('href'), location.href).href:'',"
    "snippet:(el.querySelector('.searchresult')?.textContent||'').trim()}})"
    ".filter(r=>r.title&&r.url)"
    ")"
)

_PAGE_TEXT_JS = (
    "JSON.stringify("
    "[...document.querySelectorAll('p,h1,h2,h3,li')]"
    ".map(e=>e.textContent.trim())"
    ".filter(t=>t.length>20)"
    ".slice(0,60)"
    ".join(' ')"
    ")"
)

_SEARCH_TARGETS = (
    ("google", GOOGLE_SEARCH_URL, _GOOGLE_EXTRACT_JS),
    ("yahoo", YAHOO_SEARCH_URL, _YAHOO_EXTRACT_JS),
    ("wikipedia", WIKIPEDIA_SEARCH_URL, _WIKIPEDIA_EXTRACT_JS),
)


@dataclass(frozen=True)
class BrowserSearchConfig:
    topk: int = DEFAULT_TOPK
    batch_workers: int = 4
    subprocess_timeout: int = SUBPROCESS_TIMEOUT
    fetch_content: bool = True
    content_timeout: int = 10


class BrowserSearchEngine:
    def __init__(self, config: BrowserSearchConfig):
        self.config = config

    def _run(
        self,
        *args: str,
        session: str | None = None,
        raw: bool = False,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        # --raw is a global flag; must precede -s= per playwright-cli CLI contract.
        cmd = [PLAYWRIGHT_CMD]
        if raw:
            cmd.append("--raw")
        if session:
            cmd.append(f"-s={session}")
        cmd.extend(args)
        env = os.environ.copy()
        env.setdefault("TMPDIR", "/tmp")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
                if timeout is not None
                else self.config.subprocess_timeout,
                env=env,
            )
        except FileNotFoundError as exc:
            raise BrowserSearchUnavailableError(
                f"{PLAYWRIGHT_CMD!r} is not installed or not on PATH, so the "
                f"browser search provider cannot run. It is a separate binary: "
                f"the `playwright` pip wheel does not provide it, and the "
                f"container image does not install it."
            ) from exc
        if proc.returncode != 0:
            stderr = getattr(proc, "stderr", "") or ""
            raise RuntimeError(f"{' '.join(cmd)} failed: {stderr.strip()}")
        return proc

    def _extract_hits(self, js: str, *, session: str) -> list[dict[str, str]]:
        proc = self._run("eval", js, session=session, raw=True)
        value = json.loads(proc.stdout.strip()) if proc.stdout.strip() else []
        if isinstance(value, str):
            value = json.loads(value) if value else []
        return (
            [h for h in value if isinstance(h, dict)] if isinstance(value, list) else []
        )

    def _fetch_page_text(self, url: str, *, session: str) -> str:
        try:
            self._run("goto", url, session=session, timeout=self.config.content_timeout)
            self._run("snapshot", session=session, timeout=self.config.content_timeout)
            proc = self._run(
                "eval",
                _PAGE_TEXT_JS,
                session=session,
                raw=True,
                timeout=self.config.content_timeout,
            )
            if proc.stdout.strip():
                raw = json.loads(proc.stdout.strip())
                return str(raw)[:3000] if raw else ""
        except Exception as exc:
            logger.info("fetch_page_text failed for %s: %s", url, exc)
        return ""

    def _search_and_process(self, query: str) -> list[dict[str, dict[str, str]]]:
        session = f"search-{uuid.uuid4().hex[:8]}"
        hits: list[dict[str, str]] = []
        try:
            self._run("open", "about:blank", "--persistent", session=session)
            for name, url_template, extract_js in _SEARCH_TARGETS:
                url = url_template.format(
                    query=quote_plus(query),
                    topk=max(self.config.topk, DEFAULT_TOPK),
                )
                try:
                    self._run("goto", url, session=session)
                    self._run("snapshot", session=session)
                    hits = self._extract_hits(extract_js, session=session)
                except Exception as exc:
                    logger.info(
                        "browser search target %s failed for %r: %s",
                        name,
                        query,
                        exc,
                    )
                    hits = []
                if hits:
                    logger.info(
                        "browser search target %s returned %d hit(s) for %r",
                        name,
                        len(hits),
                        query,
                    )
                    break

            results = []
            for h in hits[: self.config.topk]:
                url = h.get("url", "")
                content = h.get("snippet", "")
                if self.config.fetch_content and url:
                    fetched = self._fetch_page_text(url, session=session)
                    if fetched:
                        content = fetched
                results.append(format_document(h.get("title"), content, url or None))
            return results
        except BrowserSearchUnavailableError:
            # Not a failed search -- the provider cannot run at all. Swallowing
            # this is what made a missing binary look like an empty result set.
            raise
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
            logger.warning("browser search failed for %r: %s", query, exc)
            return []
        finally:
            try:
                self._run("close", session=session)
            except Exception:
                pass

    def batch_search(self, queries: list[str]) -> list[list[dict[str, dict[str, str]]]]:
        max_workers = min(max(len(queries), 1), self.config.batch_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._search_and_process, queries))


def create_app(config: BrowserSearchConfig) -> FastAPI:
    return create_search_app(
        "Browser Retrieval (playwright-cli)", BrowserSearchEngine(config)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browser-based retrieval server (playwright-cli)"
    )
    add_host_port_args(
        parser,
        "BROWSER_RETRIEVAL_HOST",
        "BROWSER_RETRIEVAL_PORT",
        DEFAULT_HOST,
        DEFAULT_PORT,
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    if not playwright_cli_available():
        # Checked here rather than in create_app so tests can still build the
        # app. A server whose every request shells out to a missing binary would
        # start, report healthy, and answer every query with no results.
        raise SystemExit(
            f"{PLAYWRIGHT_CMD!r} is not installed or not on PATH.\n"
            f"This server is a thin wrapper around that binary, so refusing to "
            f"start beats serving empty results. It is not a pip package: the "
            f"`playwright` wheel provides `playwright`, not {PLAYWRIGHT_CMD!r}, "
            f"and the container image does not install it -- run this on a host "
            f"that has it, and point AGENTIC_SEARCH_BROWSER_SEARCH_URL there."
        )
    config = BrowserSearchConfig(topk=args.topk, batch_workers=args.workers)
    app = create_app(config)
    run_uvicorn_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
