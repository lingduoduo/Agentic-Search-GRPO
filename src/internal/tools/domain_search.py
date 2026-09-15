"""Local domain discovery and search over the repository's existing tools."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from .base import Tool, ToolEffect
from .search import fetch_url, make_web_cascade_search
from .search_domains import (
    DOMAIN_REGISTRY,
    normalize_search_domain,
    prepare_domain_query,
)
from .validation import validate_arguments

# Route only to known public-data tools, never arbitrary tools or corpus search.
# Query parameters and option schemas are taken from the existing tool definitions.
CAPABILITY_ROUTES = {
    "general.wikipedia": ("search_wikipedia", "query"),
    "academic.arxiv": ("search_arxiv", "query"),
    "resource.wayback": ("search_wayback", "url"),
    "finance.quote": ("get_stock_quote", "symbol"),
    "finance.crypto": ("get_crypto_price", "symbol"),
    "finance.currency": ("convert_currency", "from_currency"),
    "environment.weather": ("get_weather", "location"),
    "travel.location": ("search_location", "query"),
    "travel.nearby": ("search_nearby_places", "query"),
}


def parse_search_params(value: dict | str | None) -> dict:
    """Parse structured options, including the sample's key=value aliases."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("params must be an object, JSON object, or key=value pairs")
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        raw = value.strip()
        braces = raw.startswith("{") and raw.endswith("}")
        text, separator = (raw[1:-1], ":") if braces else (raw, "=")
        result = {}
        for pair in text.split(","):
            key, found, val = pair.partition(separator)
            key, val = key.strip().strip("\"'"), val.strip()
            if not found or not key:
                raise ValueError("params must be valid JSON or key=value pairs")
            try:
                result[key] = json.loads(val)
            except json.JSONDecodeError:
                result[key] = val.strip("\"'")
    if not isinstance(result, dict):
        raise ValueError("params must be a JSON object")
    return result


class DomainSearch:
    """Shared operations for FunctionTools and Python callers; no new backend."""

    def __init__(
        self, *, web_search_fn=None, tools: Iterable[Tool] | None = None, fetch_fn=None
    ):
        if tools is None:
            from .public_data import public_data_tools

            tools = public_data_tools()
        by_name = {
            tool.name: tool for tool in tools if tool.effect == ToolEffect.READ_ONLY
        }
        self.routes = {
            tag: (by_name[name], query_parameter)
            for tag, (name, query_parameter) in CAPABILITY_ROUTES.items()
            if name in by_name
        }
        self.web_search_fn = (
            web_search_fn
            if web_search_fn is not None
            else make_web_cascade_search(
                browser_search_url=os.getenv("AGENTIC_SEARCH_BROWSER_SEARCH_URL")
            )
        )
        self.fetch_fn = fetch_fn if fetch_fn is not None else fetch_url

    def get_sub_domains(self, domains: list[str]) -> dict[str, Any]:
        """Describe implemented routes and their real parameter schemas locally."""
        if not isinstance(domains, list) or not 1 <= len(domains) <= 5:
            raise ValueError("provide one to five domains")
        directories = []
        for value in domains:
            domain = normalize_search_domain(value)
            entries = [
                {
                    "sub_domain": f"{domain}.web",
                    "tool_name": "web_search",
                    "description": f"Search the public web with the {domain} topic hint.",
                    "query_parameter": "query",
                    "query_format": "Natural-language search query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": ["query"],
                    },
                    "params": {},
                }
            ]
            for tag, (tool, query_parameter) in self.routes.items():
                if not tag.startswith(domain + "."):
                    continue
                schema = copy.deepcopy(tool.schema.parameters)
                properties = schema.get("properties", {})
                entries.append(
                    {
                        "sub_domain": tag,
                        "tool_name": tool.name,
                        "description": tool.schema.description,
                        "query_parameter": query_parameter,
                        "query_format": properties.get(query_parameter, {}).get(
                            "description", query_parameter
                        ),
                        "parameters": schema,
                        "params": {
                            name: {
                                **spec,
                                "required": name in schema.get("required", []),
                            }
                            for name, spec in properties.items()
                            if name != query_parameter
                        },
                    }
                )
            directories.append(
                {
                    "domain": domain,
                    "description": DOMAIN_REGISTRY[domain].description,
                    "sub_domains": entries,
                }
            )
        return {"domains": directories}

    async def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        tag: str | None = None,
        sub_domain: str | None = None,
        params: dict | str | None = None,
        sub_domain_params: dict | str | None = None,
        max_results: int = 5,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise ValueError("max_results must be an integer")
        max_results = max(1, min(max_results, 10))
        if tag is not None and sub_domain is not None and tag != sub_domain:
            raise ValueError("tag and sub_domain must match")
        tag = tag if tag is not None else sub_domain
        if tag is not None and (not isinstance(tag, str) or "." not in tag):
            raise ValueError("tag must be a capability returned by get_sub_domains")
        canonical = normalize_search_domain(
            domain
            if domain is not None
            else (tag.split(".", 1)[0] if tag else "general")
        )
        tag = tag if tag is not None else f"{canonical}.web"
        if not tag.startswith(canonical + "."):
            raise ValueError("domain must match the tag prefix")
        options = parse_search_params(
            params if params is not None else sub_domain_params
        )
        if (
            params is not None
            and sub_domain_params is not None
            and options != parse_search_params(sub_domain_params)
        ):
            raise ValueError("params and sub_domain_params must match")
        result = {"query": query, "domain": canonical, "tag": tag}
        if tag == f"{canonical}.web":
            if options:
                raise ValueError(
                    "web capabilities accept max_results, not capability params"
                )
            executed_query = prepare_domain_query(query, canonical)
            pages = await self.web_search_fn(executed_query, page_size=max_results)
            errors = [page.error for page in pages if page.error]
            if errors:
                raise ValueError("; ".join(errors))
            result.update(
                executed_query=executed_query,
                results=[
                    {"title": page.title, "content": page.summary, "url": page.url}
                    for page in pages[:max_results]
                ],
            )
            return result
        if tag not in self.routes:
            raise ValueError(f"unsupported capability {tag!r}; use get_sub_domains")
        tool, query_parameter = self.routes[tag]
        properties = tool.schema.parameters.get("properties", {})
        unknown = options.keys() - properties.keys()
        if unknown:
            raise ValueError(
                "unsupported capability params: " + ", ".join(sorted(unknown))
            )
        if query_parameter in options and options[query_parameter] != query:
            raise ValueError(f"query and params.{query_parameter} must match")
        arguments = {**options, query_parameter: query}
        if "limit" in properties and "limit" not in arguments:
            arguments["limit"] = max_results
        errors = validate_arguments(tool.schema.parameters, arguments)
        if errors:
            raise ValueError("; ".join(errors))
        if "limit" in arguments:
            arguments["limit"] = max(1, min(arguments["limit"], max_results))
        instance_id = await tool.create()
        try:
            response, _, _ = await tool.execute(instance_id, arguments)
        finally:
            await tool.release(instance_id)
        payload = json.loads(response)
        if isinstance(payload, dict) and "error" in payload:
            raise ValueError(str(payload["error"]))
        if not isinstance(payload, (list, dict)):
            raise ValueError("capability returned an unsupported result")
        result["results"] = (
            payload[:max_results] if isinstance(payload, list) else payload
        )
        return result

    async def extract(
        self, url: str, *, max_length: int = 5000
    ) -> list[dict[str, str]]:
        if (
            not isinstance(url, str)
            or urlsplit(url).scheme not in ("http", "https")
            or not urlsplit(url).netloc
        ):
            raise ValueError("url must be an HTTP(S) URL")
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or not 1 <= max_length <= 50000
        ):
            raise ValueError("max_length must be between 1 and 50000")
        content = await self.fetch_fn(url, max_length=max_length)
        if content.startswith("[fetch error]"):
            raise ValueError(content)
        return [{"title": url, "content": content, "url": url}]

    async def batch_search(self, queries: list[dict], **shared_options) -> list[dict]:
        if not isinstance(queries, list) or not 1 <= len(queries) <= 5:
            raise ValueError("batch_search supports one to five queries")

        async def run(item):
            try:
                if not isinstance(item, dict):
                    raise ValueError("each query must be an object")
                defaults = dict(shared_options)
                if item.get("tag") or item.get("sub_domain"):
                    for key in ("tag", "domain", "sub_domain"):
                        defaults.pop(key, None)
                elif "domain" in item:
                    for key in ("tag", "sub_domain"):
                        defaults.pop(key, None)
                if "params" in item or "sub_domain_params" in item:
                    defaults.pop("params", None)
                    defaults.pop("sub_domain_params", None)
                return await self.search(**{**defaults, **item})
            except Exception as error:
                return {
                    "query": item.get("query", "") if isinstance(item, dict) else "",
                    "error": str(error),
                }

        # Parent cancellation propagates; every normal per-item failure is retained.
        return await asyncio.gather(*(run(item) for item in queries))
