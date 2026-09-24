"""Reference-lookup tools: Wikipedia, ArXiv, and the Wayback Machine.

These three are the citeable ones: each returns a JSON array of
``{"title", "content", "url"}``, the same shape the corpus search returns, so
their results become source cards.
"""

from __future__ import annotations

import re
from datetime import datetime
from xml.etree import ElementTree

from ..base import FunctionTool, ToolEffect
from ._http import MAX_CONTENT_CHARS, PublicDataError, get_json, get_text, guarded

WIKIPEDIA_API = "https://{language}.wikipedia.org/w/api.php"
ARXIV_API = "https://export.arxiv.org/api/query"
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"

_ATOM = "{http://www.w3.org/2005/Atom}"
# The language code is interpolated into a hostname, so it is validated rather
# than escaped: anything outside this shape could redirect the request.
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(-[a-z]{2,8})?$")
_ARXIV_SORTS = {"relevance", "lastUpdatedDate", "submittedDate"}


# ---------------------------------------------------------------- Wikipedia

_WIKIPEDIA_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "description": "What to look up."},
        "language": {
            "type": "string",
            "pattern": "^[A-Za-z]+(-[A-Za-z]+)*$",
            "maxLength": 20,
            "description": "Wikipedia language code, e.g. 'en' or 'de'.",
            "default": "en",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "How many articles to return (1-10).",
            "default": 3,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def _search_wikipedia(
    query: str, language: str = "en", limit: int = 3
) -> list[dict]:
    language = language.strip().lower()
    if not _LANGUAGE_RE.match(language):
        raise PublicDataError(f"unsupported Wikipedia language code {language!r}")
    api = WIKIPEDIA_API.format(language=language)
    limit = max(1, min(int(limit), 10))

    found = await get_json(
        api,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        },
    )
    hits = (found.get("query") or {}).get("search") or []
    if not hits:
        return []

    page_ids = [str(hit["pageid"]) for hit in hits if hit.get("pageid")]
    extracts = await get_json(
        api,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "pageids": "|".join(page_ids),
            "format": "json",
        },
    )
    pages = ((extracts.get("query") or {}).get("pages")) or {}
    results = []
    for page_id in page_ids:
        page = pages.get(page_id) or {}
        title = page.get("title", "")
        if not title:
            # A page id missing from the extracts response has nothing worth
            # showing — skip it rather than emit a blank source card.
            continue
        results.append(
            {
                "title": title,
                "content": (page.get("extract") or "")[:MAX_CONTENT_CHARS],
                # ?curid= is the stable per-page URL and needs no title escaping.
                "url": f"https://{language}.wikipedia.org/?curid={page_id}",
            }
        )
    return results


def build_wikipedia_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_search_wikipedia),
        name="search_wikipedia",
        description=(
            "Look up encyclopedia articles on Wikipedia. Use for background on "
            "people, places, organizations, events, and general concepts."
        ),
        parameters=_WIKIPEDIA_PARAMS,
        effect=ToolEffect.READ_ONLY,
        citeable=True,
    )


# -------------------------------------------------------------------- ArXiv

_ARXIV_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "description": "Topic, title, or author.",
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 25,
            "description": "How many papers to return (1-25).",
            "default": 3,
        },
        "sort_by": {
            "type": "string",
            "enum": sorted(_ARXIV_SORTS),
            "description": "One of relevance, lastUpdatedDate, submittedDate.",
            "default": "relevance",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def _search_arxiv(
    query: str, max_results: int = 3, sort_by: str = "relevance"
) -> list[dict]:
    body = await get_text(
        ARXIV_API,
        params={
            "search_query": f"all:{query}",
            "max_results": max(1, min(int(max_results), 25)),
            "sortBy": sort_by if sort_by in _ARXIV_SORTS else "relevance",
            "sortOrder": "descending",
        },
    )
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise PublicDataError(
            "arxiv returned a malformed Atom feed", upstream=True
        ) from exc

    results = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = entry.findtext(f"{_ATOM}title") or ""
        summary = entry.findtext(f"{_ATOM}summary") or ""
        results.append(
            {
                # Atom wraps titles across lines; collapse the whitespace.
                "title": " ".join(title.split()),
                "content": " ".join(summary.split())[:MAX_CONTENT_CHARS],
                "url": (entry.findtext(f"{_ATOM}id") or "").strip(),
            }
        )
    return results


def build_arxiv_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_search_arxiv),
        name="search_arxiv",
        description=(
            "Search ArXiv for academic preprints and return their abstracts. "
            "Use for research papers, methods, and scientific results."
        ),
        parameters=_ARXIV_PARAMS,
        effect=ToolEffect.READ_ONLY,
        citeable=True,
    )


# ----------------------------------------------------------------- Wayback

_WAYBACK_PARAMS = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "minLength": 1, "description": "The URL to look up."},
        "year": {
            "type": "integer",
            # The Wayback Machine began in 1996; CDX timestamps carry 4-digit years.
            "minimum": 1996,
            "maximum": 9999,
            "description": "Restrict snapshots to this calendar year.",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "description": "How many snapshots to return (1-50).",
            "default": 10,
        },
    },
    "required": ["url"],
    "additionalProperties": False,
}


async def _search_wayback(
    url: str, year: int | None = None, limit: int = 10
) -> list[dict]:
    params = {
        "url": url,
        "output": "json",
        "limit": max(1, min(int(limit), 50)),
        "fl": "timestamp,original,statuscode,mimetype",
    }
    if year:
        params["from"] = f"{int(year)}0101"
        params["to"] = f"{int(year)}1231"

    rows = await get_json(WAYBACK_CDX, params=params)
    # The CDX API answers with a header row followed by data rows; a lone
    # header row means "archived nothing".
    if not isinstance(rows, list) or len(rows) < 2:
        return []

    headers, *data = rows
    results = []
    for row in data:
        record = dict(zip(headers, row))
        stamp = record.get("timestamp", "")
        try:
            when = datetime.strptime(stamp, "%Y%m%d%H%M%S").isoformat()
        except ValueError:
            when = stamp
        original = record.get("original", url)
        results.append(
            {
                "title": f"Snapshot of {original} at {when}",
                "content": (
                    f"HTTP {record.get('statuscode', '?')}, "
                    f"{record.get('mimetype', 'unknown')}"
                ),
                "url": f"https://web.archive.org/web/{stamp}/{original}",
            }
        )
    return results


def build_wayback_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_search_wayback),
        name="search_wayback",
        description=(
            "Find archived snapshots of a web page in the Internet Archive's "
            "Wayback Machine. Use to see what a URL looked like in the past."
        ),
        parameters=_WAYBACK_PARAMS,
        effect=ToolEffect.READ_ONLY,
        citeable=True,
    )
