# Wire the /api/agent browser fallback URL

**Gap.** `/api/agent` auto-search has a browser leg after SerpAPI (`app.py`, `if browser_search_url`). But `SearchExperienceSettings.from_app_settings()` never set `browser_search_url`, so the leg could not be reached under the default construction. `AGENTIC_SEARCH_BROWSER_SEARCH_URL`, which CLAUDE.md tells operators to set on the web backend, reached only the `web_search` tool. It is the same "built but unreachable" shape as the rerank URL before #431.

**Fix.** Mirror #431: `ServiceSettings.browser_search_url` is read from `AGENTIC_SEARCH_BROWSER_SEARCH_URL` and passed through `from_app_settings()`. When the variable is unset, nothing changes.

**Behaviour change.** An operator who already set the variable for the `web_search` tool now also gets the browser leg on `/api/agent`, but only after SerpAPI yields nothing. That leg is slow (~30-50 s) and sits behind the existing browser circuit breaker (#649).
