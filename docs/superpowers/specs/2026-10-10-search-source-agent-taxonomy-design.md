# Search-source agent taxonomy migration

## Intent

Replace the existing sixteen topic domains plus general with the user-provided twenty search-source agent categories. API discovery, tool schemas, capability directories, and the Assist selector must share one canonical registry. Topics such as health, energy, security, and gaming remain query subjects rather than selectable source categories. The user explicitly selected compatibility aliases for existing clients.

## Canonical taxonomy

| Agent | Primary search topics | Typical sources |
| --- | --- | --- |
| web | General information, websites, explanations, references | Web search engines |
| academic | Papers, preprints, citations, conferences, research | arXiv, Semantic Scholar, OpenAlex |
| code | Repositories, APIs, libraries, examples, issues | GitHub, GitLab, package registries |
| social_media | Discussions, opinions, trends, community experiences | Reddit, Hacker News, public social platforms |
| resource | Datasets, directories, tools, templates, benchmarks | Hugging Face, Kaggle, open-data portals |
| news | Breaking news, reporting, investigations, press releases | News publishers, news aggregators |
| official | Official announcements, documentation, authoritative statements | Government agencies, organizations, official websites |
| legal | Statutes, regulations, case law, legal decisions | Court databases, legal repositories |
| financial | Market data, filings, earnings, financial statements | SEC EDGAR, exchange data, financial providers |
| patent | Patents, trademarks, inventions, intellectual property | Google Patents, USPTO, WIPO |
| product | Product catalogs, specifications, prices, reviews | Retailers, manufacturers, shopping indexes |
| business | Company profiles, industry information, organizations | Company registries, corporate websites |
| local | Businesses, locations, hours, local services | Maps, business directories |
| travel | Flights, hotels, destinations, transportation | Travel providers, tourism websites |
| media | Images, videos, audio, podcasts, multimedia | YouTube, Wikimedia Commons, media platforms |
| books | Books, authors, editions, publications, libraries | Open Library, Google Books, library catalogs |
| education | Courses, tutorials, textbooks, educational programs | University websites, course catalogs, learning platforms |
| events | Conferences, concerts, meetups, schedules | Event platforms, official event sites |
| jobs | Job openings, careers, salaries, employment opportunities | Employer career pages, job boards |
| statistics | Public statistics, indicators, surveys, census data | Census Bureau, World Bank, OECD |

## Registry and behavior

Keep the existing domain field and endpoint names to avoid an unrelated API rename. Canonical discovery exposes exactly these twenty agents in table order. Store primary topics in the description and typical sources as immutable registry metadata; expose typical_sources as an additive array in discovery and capability directories. Update frontend types and selection descriptions accordingly. Default omitted selectors and UI fallback to web, whose query hint is empty.

Each category retains the implemented web capability and a concise source-oriented query hint. Hints guide web search, not guaranteed provider selection or hard result filtering. Typical sources describe the intended source family, not an assertion that dedicated connectors exist. Preserve provider selection, authorization filters, timeout and error behavior. This migration does not introduce twenty autonomous agent instances, new provider integrations, automatic multi-agent routing, or fan-out search.

## Compatibility

Normalize case, whitespace, hyphens as today, then aliases: general -> web; finance -> financial; ip -> patent; health, security, energy, environment, agriculture, gaming -> web; film -> media. Existing identifiers that are canonical retain their meaning. Responses use canonical identifiers. Removed topic selectors remain accepted but no longer append their old subject hints; the caller's query carries its subject.

Canonicalize legacy capability-tag prefixes before validating domain/tag agreement. Preserve old public-data routes: general.wikipedia -> web.wikipedia; finance.quote/crypto/currency -> financial equivalents; environment.weather -> local.weather. Preserve academic.arxiv and resource.wayback. Keep travel.location/nearby for compatibility and also expose these implemented location capabilities under local. Old environment.web resolves to web.web, while environment.weather uses the explicit tag migration. Reject unknown selectors, unknown tags, and conflicting canonical domain/tag pairs before dispatch.

Alias normalization must apply consistently across direct tools, MCP, HTTP input validation, and capability discovery. Schemas and menus advertise canonical names only, while input validators explicitly permit legacy values. Do not let JSON-schema validation reject aliases before normalization.

## Validation and documentation

Add tests for exact ordered twenty-category discovery, metadata, web defaults, all alias mappings, legacy public-data tags, tag/domain consistency, invalid inputs, web query hinting, HTTP mode validation, MCP dispatch, and Assist fallback/selection. Verify real public-data tool reuse without network calls. Update active README/search documentation and current evaluation fixtures/workflows so removed identifiers do not cause drift. Preserve historical specs and plans as historical records. Evaluation labels may use canonical source categories only when their relevance labels still match; otherwise label legacy fixtures explicitly rather than inventing benchmark evidence.

Run affected Python and frontend tests, lint/type checks, shared search-query consumers, and relevant policy guards. Request one independent whole-branch review, create a separate PR against main, and verify CI for its final commit. Do not merge.

## Review focus

Check default web behavior through every entry point; schema acceptance of aliases; environment.weather versus environment.web disambiguation; capability routes invoking the same tools with the same arguments; additive discovery metadata compatibility; frontend fallback on discovery failure; and evaluation fixtures silently changing interpretation. No live-source coverage or quality improvement is claimed.
