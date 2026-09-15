# Domain Relevance Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether selecting a search domain moves web results toward that topic's authoritative sources, closing the evaluation the taxonomy spec has required since it shipped.

**Architecture:** A torch-free measurement core in `src/internal/retrieval/domain_eval.py` runs each labeled query through `search_tool` twice — once raw, once hinted — and scores each arm by the share of top-10 results from that query's authority hosts. A CLI in `examples/` applies a paired permutation test over the per-query deltas and writes a report to `data/eval/`. Every provider response is cached so re-runs cost no quota.

**Tech Stack:** Python 3.10+, numpy, pytest. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-09-15-domain-relevance-eval-design.md`

## Global Constraints

- Python >=3.10; no new runtime dependencies.
- `src/internal/retrieval/domain_eval.py` must stay torch-free: it must never import from `src/model/post_training`, whose package `__init__` side effects pull torch in.
- Top-k is fixed at k=10.
- The unit of analysis is the query, never the individual result.
- `general` is a negative control and is excluded from the paired deltas.
- Per-domain results are descriptive only; no per-domain p-values are computed or printed.
- Unit tests never open a network socket.
- `data/` is gitignored: committed data files require `git add -f`.
- Report floats must be finite; the Dev Console reads `data/eval/*.json` and a non-finite float reaches the frontend as `null`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/internal/retrieval/domain_eval.py` | Host matching, authority precision, jaccard, the two-arm runner, the disk cache |
| `data/eval/domain_relevance_queries.json` | The label set: 16 topic domains x 3 queries, with authority hosts |
| `examples/run_domain_relevance_eval.py` | CLI: loads labels, runs the core, applies paired statistics, writes the report |
| `tests/unit/test_domain_eval.py` | All offline tests, stub provider |
| `docs/search-engine.md` | Document how to run the evaluation and how to read its limits |

---

### Task 1: Scoring primitives

**Files:**
- Create: `src/internal/retrieval/domain_eval.py`
- Test: `tests/unit/test_domain_eval.py`

**Interfaces:**
- Produces: `host_matches(host: str, pattern: str) -> bool`; `authority_precision(urls: list[str], hosts: set[str]) -> float`; `jaccard(a: list[str], b: list[str]) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_domain_eval.py`:

```python
"""Offline tests for the domain relevance evaluation.

No test in this file opens a socket: the runner is driven with a stub
provider so the suite stays fast and spends no SerpAPI quota.
"""

from __future__ import annotations

import pytest

from src.internal.retrieval.domain_eval import (
    authority_precision,
    host_matches,
    jaccard,
)


@pytest.mark.parametrize(
    "host,pattern,expected",
    [
        ("stanford.edu", "stanford.edu", True),
        ("cs.stanford.edu", "stanford.edu", True),
        # A suffix collision is not a subdomain: the boundary is a dot.
        ("notstanford.edu", "stanford.edu", False),
        ("stanford.edu.evil.com", "stanford.edu", False),
        ("ARXIV.ORG", "arxiv.org", True),
        ("", "arxiv.org", False),
    ],
)
def test_host_matches_requires_a_dot_boundary(host, pattern, expected):
    assert host_matches(host, pattern) is expected


def test_authority_precision_all_and_none():
    hosts = {"arxiv.org"}
    assert authority_precision(["https://arxiv.org/abs/1", "http://arxiv.org/x"], hosts) == 1.0
    assert authority_precision(["https://example.com/a"], hosts) == 0.0


def test_authority_precision_is_a_share_of_results():
    hosts = {"arxiv.org"}
    urls = ["https://arxiv.org/abs/1", "https://example.com/a", "https://b.com/c", "https://d.com"]
    assert authority_precision(urls, hosts) == 0.25


def test_authority_precision_of_no_results_is_zero():
    # An empty arm scores zero rather than raising, so an empty provider
    # response and a wholly irrelevant one are comparable.
    assert authority_precision([], {"arxiv.org"}) == 0.0


def test_authority_precision_ignores_unparseable_urls():
    assert authority_precision(["not a url", "https://arxiv.org/x"], {"arxiv.org"}) == 0.5


def test_jaccard_identical_and_disjoint():
    assert jaccard(["a", "b"], ["b", "a"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0


def test_jaccard_of_two_empty_arms_is_one():
    # Two arms that both returned nothing are identical, not undefined.
    assert jaccard([], []) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_domain_eval.py -q`
Expected: FAIL — `No module named 'src.internal.retrieval.domain_eval'`.

- [ ] **Step 3: Implement the primitives**

Create `src/internal/retrieval/domain_eval.py`:

```python
"""Measure whether a domain hint moves web results toward topic sources.

This module stays torch-free on purpose: importing
``src.model.post_training`` pulls torch in through package ``__init__``
side effects, which would drop this module out of the torch-free CI job.
The paired statistics therefore live in the CLI, not here.

What it measures is source alignment, not answer quality. A result from
arxiv.org is not automatically a better answer than a good blog post.
"""

from __future__ import annotations

from urllib.parse import urlparse

TOP_K = 10


def host_matches(host: str, pattern: str) -> bool:
    """True when *host* is *pattern* or a subdomain of it.

    The boundary is a dot, so "notstanford.edu" does not match
    "stanford.edu" and "stanford.edu.evil.com" does not either.
    """
    host = (host or "").strip().lower()
    pattern = (pattern or "").strip().lower()
    if not host or not pattern:
        return False
    return host == pattern or host.endswith("." + pattern)


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def authority_precision(urls: list[str], hosts: set[str]) -> float:
    """Share of *urls* served by one of the authority *hosts*.

    An empty list scores 0.0 rather than raising, so an arm that returned
    nothing is comparable with one that returned only off-topic results.
    """
    if not urls:
        return 0.0
    hits = sum(
        1 for url in urls if any(host_matches(_host_of(url), h) for h in hosts)
    )
    return hits / len(urls)


def jaccard(a: list[str], b: list[str]) -> float:
    """Overlap of two result URL lists. Two empty arms are identical."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_domain_eval.py -q`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/domain_eval.py tests/unit/test_domain_eval.py
git commit -m "feat(eval): add scoring primitives for domain relevance"
```

---

### Task 2: The label set

**Files:**
- Create: `data/eval/domain_relevance_queries.json`
- Modify: `src/internal/retrieval/domain_eval.py`
- Test: `tests/unit/test_domain_eval.py`

**Interfaces:**
- Consumes: `AVAILABLE_DOMAINS`, `DOMAIN_REGISTRY` from `src.internal.tools.search`
- Produces: `LabelledQuery` dataclass with fields `domain: str`, `query: str`, `authority_hosts: set[str]`; `load_labels(path: str | Path) -> list[LabelledQuery]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_domain_eval.py`:

```python
from pathlib import Path

from src.internal.retrieval.domain_eval import DEFAULT_LABELS_PATH, load_labels
from src.internal.tools.search import AVAILABLE_DOMAINS


def test_labels_cover_every_topic_domain():
    # The taxonomy and the evaluation must not drift apart. `general` has an
    # empty hint and is carried separately as a control.
    labels = load_labels(DEFAULT_LABELS_PATH)
    covered = {lab.domain for lab in labels}
    expected = {d for d in AVAILABLE_DOMAINS if d != "general"}
    assert covered == expected


def test_every_domain_has_three_queries():
    labels = load_labels(DEFAULT_LABELS_PATH)
    counts = {}
    for lab in labels:
        counts[lab.domain] = counts.get(lab.domain, 0) + 1
    assert set(counts.values()) == {3}, counts


def test_every_query_carries_authority_hosts():
    for lab in load_labels(DEFAULT_LABELS_PATH):
        assert lab.authority_hosts, f"{lab.domain}: {lab.query}"


def test_missing_label_file_fails_loudly(tmp_path):
    # Failing before any provider call beats silently evaluating fewer domains.
    with pytest.raises(FileNotFoundError) as exc:
        load_labels(tmp_path / "absent.json")
    assert "absent.json" in str(exc.value)


def test_malformed_label_file_names_the_path(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError) as exc:
        load_labels(bad)
    assert "bad.json" in str(exc.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_domain_eval.py -q -k label`
Expected: FAIL — `load_labels` does not exist.

- [ ] **Step 3: Write the label file**

Create `data/eval/domain_relevance_queries.json`. Three queries per topic
domain, each with the hosts an on-topic result should come from. Two queries
per domain are plainly in-topic; the third is deliberately ambiguous across
domains, because the spec asks for overlapping topics.

```json
{
  "k": 10,
  "domains": {
    "resource": [
      {"query": "open datasets for image classification", "authority_hosts": ["kaggle.com", "huggingface.co", "data.gov", "zenodo.org"]},
      {"query": "public api directory", "authority_hosts": ["github.com", "publicapis.dev", "rapidapi.com", "programmableweb.com"]},
      {"query": "climate data archive", "authority_hosts": ["noaa.gov", "data.gov", "copernicus.eu", "zenodo.org"]}
    ],
    "social_media": [
      {"query": "what people are saying about remote work", "authority_hosts": ["reddit.com", "x.com", "twitter.com", "linkedin.com", "threads.net"]},
      {"query": "community discussion mechanical keyboards", "authority_hosts": ["reddit.com", "discord.com", "news.ycombinator.com", "quora.com"]},
      {"query": "influencer marketing engagement rates", "authority_hosts": ["instagram.com", "tiktok.com", "linkedin.com", "hootsuite.com", "sproutsocial.com"]}
    ],
    "finance": [
      {"query": "etf expense ratio comparison", "authority_hosts": ["morningstar.com", "bloomberg.com", "investopedia.com", "sec.gov", "ft.com"]},
      {"query": "federal reserve interest rate decision", "authority_hosts": ["federalreserve.gov", "reuters.com", "bloomberg.com", "wsj.com", "ft.com"]},
      {"query": "carbon credit pricing", "authority_hosts": ["bloomberg.com", "reuters.com", "spglobal.com", "icapcarbonaction.com"]}
    ],
    "academic": [
      {"query": "transformer attention mechanism", "authority_hosts": ["arxiv.org", "doi.org", "acm.org", "ieee.org", "openreview.net"]},
      {"query": "replication crisis in psychology", "authority_hosts": ["doi.org", "nature.com", "science.org", "apa.org", "pubmed.ncbi.nlm.nih.gov"]},
      {"query": "soil carbon sequestration measurement", "authority_hosts": ["doi.org", "nature.com", "arxiv.org", "sciencedirect.com", "usda.gov"]}
    ],
    "legal": [
      {"query": "fair use four factor test", "authority_hosts": ["law.cornell.edu", "copyright.gov", "supremecourt.gov", "courtlistener.com", "justia.com"]},
      {"query": "gdpr data subject access request deadline", "authority_hosts": ["gdpr-info.eu", "edpb.europa.eu", "ico.org.uk", "eur-lex.europa.eu"]},
      {"query": "employee non compete enforceability", "authority_hosts": ["law.cornell.edu", "ftc.gov", "justia.com", "nolo.com", "courtlistener.com"]}
    ],
    "health": [
      {"query": "statin side effects evidence", "authority_hosts": ["pubmed.ncbi.nlm.nih.gov", "nih.gov", "cdc.gov", "mayoclinic.org", "nhs.uk", "cochrane.org"]},
      {"query": "measles vaccination schedule", "authority_hosts": ["cdc.gov", "who.int", "nhs.uk", "aap.org", "nih.gov"]},
      {"query": "air pollution respiratory outcomes", "authority_hosts": ["who.int", "epa.gov", "pubmed.ncbi.nlm.nih.gov", "thelancet.com", "nih.gov"]}
    ],
    "business": [
      {"query": "saas net revenue retention benchmarks", "authority_hosts": ["hbr.org", "mckinsey.com", "bain.com", "gartner.com", "bvp.com"]},
      {"query": "supply chain resilience strategy", "authority_hosts": ["mckinsey.com", "hbr.org", "deloitte.com", "bcg.com", "gartner.com"]},
      {"query": "carbon accounting for corporates", "authority_hosts": ["ghgprotocol.org", "cdp.net", "deloitte.com", "mckinsey.com", "sciencebasedtargets.org"]}
    ],
    "security": [
      {"query": "log4shell mitigation guidance", "authority_hosts": ["cisa.gov", "nvd.nist.gov", "mitre.org", "apache.org", "sans.org"]},
      {"query": "owasp top ten injection", "authority_hosts": ["owasp.org", "cisa.gov", "portswigger.net", "mitre.org", "nist.gov"]},
      {"query": "ransomware incident response plan", "authority_hosts": ["cisa.gov", "nist.gov", "sans.org", "fbi.gov", "ncsc.gov.uk"]}
    ],
    "ip": [
      {"query": "software patent eligibility alice test", "authority_hosts": ["uspto.gov", "law.cornell.edu", "supremecourt.gov", "wipo.int", "justia.com"]},
      {"query": "trademark opposition procedure", "authority_hosts": ["uspto.gov", "wipo.int", "euipo.europa.eu", "inta.org"]},
      {"query": "open source license compatibility", "authority_hosts": ["opensource.org", "gnu.org", "fsf.org", "apache.org", "choosealicense.com"]}
    ],
    "code": [
      {"query": "python asyncio gather exception handling", "authority_hosts": ["docs.python.org", "stackoverflow.com", "github.com", "realpython.com"]},
      {"query": "rust borrow checker lifetime error", "authority_hosts": ["doc.rust-lang.org", "stackoverflow.com", "github.com", "rust-lang.org"]},
      {"query": "postgres index performance tuning", "authority_hosts": ["postgresql.org", "stackoverflow.com", "github.com", "percona.com", "citusdata.com"]}
    ],
    "energy": [
      {"query": "grid scale battery storage cost", "authority_hosts": ["nrel.gov", "iea.org", "energy.gov", "eia.gov", "bnef.com"]},
      {"query": "small modular reactor licensing", "authority_hosts": ["nrc.gov", "iaea.org", "energy.gov", "world-nuclear.org"]},
      {"query": "green hydrogen production cost", "authority_hosts": ["iea.org", "nrel.gov", "energy.gov", "irena.org"]}
    ],
    "environment": [
      {"query": "ocean acidification coral reefs", "authority_hosts": ["noaa.gov", "epa.gov", "nature.com", "ipcc.ch", "unep.org"]},
      {"query": "microplastics in drinking water", "authority_hosts": ["who.int", "epa.gov", "nih.gov", "unep.org", "nature.com"]},
      {"query": "peatland restoration carbon", "authority_hosts": ["ipcc.ch", "unep.org", "iucn.org", "nature.com", "gov.uk"]}
    ],
    "agriculture": [
      {"query": "cover crop nitrogen fixation rates", "authority_hosts": ["usda.gov", "extension.org", "fao.org", "sare.org"]},
      {"query": "integrated pest management brassica", "authority_hosts": ["usda.gov", "extension.org", "fao.org", "epa.gov"]},
      {"query": "drought tolerant maize varieties", "authority_hosts": ["cgiar.org", "fao.org", "usda.gov", "cimmyt.org"]}
    ],
    "travel": [
      {"query": "schengen visa transit rules", "authority_hosts": ["schengenvisainfo.com", "europa.eu", "gov.uk", "travel.state.gov"]},
      {"query": "best time to visit patagonia", "authority_hosts": ["lonelyplanet.com", "tripadvisor.com", "nationalgeographic.com", "roughguides.com"]},
      {"query": "night train routes across europe", "authority_hosts": ["seat61.com", "raileurope.com", "interrail.eu", "lonelyplanet.com"]}
    ],
    "film": [
      {"query": "anamorphic lens cinematography", "authority_hosts": ["imdb.com", "ascmag.com", "nofilmschool.com", "premiumbeat.com", "bfi.org.uk"]},
      {"query": "cannes palme d'or winners", "authority_hosts": ["imdb.com", "festival-cannes.com", "bfi.org.uk", "variety.com", "hollywoodreporter.com"]},
      {"query": "film financing tax incentives", "authority_hosts": ["variety.com", "hollywoodreporter.com", "bfi.org.uk", "imdb.com"]}
    ],
    "gaming": [
      {"query": "unreal engine nanite performance", "authority_hosts": ["unrealengine.com", "gamedeveloper.com", "github.com", "reddit.com", "docs.unrealengine.com"]},
      {"query": "speedrun glitch categories", "authority_hosts": ["speedrun.com", "reddit.com", "youtube.com", "twitch.tv"]},
      {"query": "game accessibility guidelines", "authority_hosts": ["gameaccessibilityguidelines.com", "igda.org", "gamedeveloper.com", "xbox.com"]}
    ]
  }
}
```

- [ ] **Step 4: Implement the loader**

Append to `src/internal/retrieval/domain_eval.py`:

```python
import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LABELS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "eval" / "domain_relevance_queries.json"
)


@dataclass(frozen=True)
class LabelledQuery:
    domain: str
    query: str
    authority_hosts: set[str] = field(default_factory=set)


def load_labels(path: str | Path = DEFAULT_LABELS_PATH) -> list[LabelledQuery]:
    """Read the label set, failing loudly before any provider call."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"label file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"label file is not valid JSON: {path}") from exc
    out: list[LabelledQuery] = []
    for domain, entries in raw.get("domains", {}).items():
        for entry in entries:
            out.append(
                LabelledQuery(
                    domain=domain,
                    query=entry["query"],
                    authority_hosts=set(entry["authority_hosts"]),
                )
            )
    return out
```

Confirm `parents[3]` resolves to the repository root before moving on: the file
is at `src/internal/retrieval/domain_eval.py`, so parents[0] is `retrieval`,
parents[1] `internal`, parents[2] `src`, parents[3] the root.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_domain_eval.py -q`
Expected: all pass.

- [ ] **Step 6: Commit, force-adding the data file**

```bash
git add -f data/eval/domain_relevance_queries.json
git add src/internal/retrieval/domain_eval.py tests/unit/test_domain_eval.py
git commit -m "feat(eval): add the labelled query set for domain relevance"
```

---

### Task 3: Cached two-arm runner

**Files:**
- Modify: `src/internal/retrieval/domain_eval.py`
- Test: `tests/unit/test_domain_eval.py`

**Interfaces:**
- Consumes: `search_tool` from `src.internal.tools.search`; `prepare_domain_query` from the same module
- Produces: `ArmResult` (fields `urls: list[str]`, `empty: bool`, `error: str | None`); `QueryOutcome` (fields `domain`, `query`, `general: ArmResult`, `domain_arm: ArmResult`, `delta: float`, `jaccard: float`, `excluded: bool`); `async run_query(label, *, search_fn, cache) -> QueryOutcome`; `DiskCache` with `get(key)` / `set(key, value)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_domain_eval.py`:

```python
from src.internal.retrieval.domain_eval import (
    DiskCache,
    LabelledQuery,
    run_query,
)


class _StubSearch:
    """Stands in for search_tool; records every query it is asked for."""

    def __init__(self, by_query: dict[str, list[str]], fail_on: str | None = None):
        self.by_query = by_query
        self.fail_on = fail_on
        self.calls: list[str] = []

    async def __call__(self, query, **_kwargs):
        self.calls.append(query)
        if self.fail_on is not None and query == self.fail_on:
            raise RuntimeError("provider exploded")
        return [_Page(u) for u in self.by_query.get(query, [])]


class _Page:
    def __init__(self, url):
        self.url = url
        self.error = None


@pytest.mark.asyncio
async def test_run_query_scores_both_arms(tmp_path):
    label = LabelledQuery("academic", "attention mechanism", {"arxiv.org"})
    stub = _StubSearch(
        {
            "attention mechanism": ["https://blog.com/a", "https://b.com/c"],
            "attention mechanism academic research": [
                "https://arxiv.org/abs/1",
                "https://b.com/c",
            ],
        }
    )
    out = await run_query(label, search_fn=stub, cache=DiskCache(tmp_path))
    assert out.general.urls == ["https://blog.com/a", "https://b.com/c"]
    assert out.delta == pytest.approx(0.5)
    assert out.jaccard == pytest.approx(1 / 3)
    assert out.excluded is False


@pytest.mark.asyncio
async def test_a_failed_arm_excludes_the_query(tmp_path):
    # A delta against a failed call measures the failure, not the domain.
    label = LabelledQuery("academic", "q", {"arxiv.org"})
    stub = _StubSearch({"q": ["https://arxiv.org/a"]}, fail_on="q academic research")
    out = await run_query(label, search_fn=stub, cache=DiskCache(tmp_path))
    assert out.excluded is True
    assert out.domain_arm.error is not None


@pytest.mark.asyncio
async def test_an_empty_arm_excludes_the_query(tmp_path):
    label = LabelledQuery("academic", "q", {"arxiv.org"})
    stub = _StubSearch({"q": ["https://arxiv.org/a"], "q academic research": []})
    out = await run_query(label, search_fn=stub, cache=DiskCache(tmp_path))
    assert out.domain_arm.empty is True
    assert out.excluded is True


@pytest.mark.asyncio
async def test_the_cache_prevents_a_second_provider_call(tmp_path):
    label = LabelledQuery("academic", "q", {"arxiv.org"})
    cache = DiskCache(tmp_path)
    stub = _StubSearch({"q": ["https://arxiv.org/a"], "q academic research": ["https://arxiv.org/b"]})
    await run_query(label, search_fn=stub, cache=cache)
    assert len(stub.calls) == 2
    await run_query(label, search_fn=stub, cache=cache)
    assert len(stub.calls) == 2, "second run must be served from cache"


@pytest.mark.asyncio
async def test_general_control_pairs_identical_arms(tmp_path):
    # An empty hint means both arms issue the same query, so the second is a
    # cache hit and the delta is structurally zero.
    label = LabelledQuery("general", "q", {"arxiv.org"})
    cache = DiskCache(tmp_path)
    stub = _StubSearch({"q": ["https://arxiv.org/a"]})
    out = await run_query(label, search_fn=stub, cache=cache)
    assert len(stub.calls) == 1
    assert out.delta == 0.0
    assert out.jaccard == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_domain_eval.py -q -k "run_query or cache or control"`
Expected: FAIL — `DiskCache` and `run_query` do not exist.

- [ ] **Step 3: Implement the cache and runner**

Append to `src/internal/retrieval/domain_eval.py`:

```python
import hashlib

from src.internal.tools.search import prepare_domain_query


class DiskCache:
    """Cache provider responses so a re-run spends no quota."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> list[str] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: list[str]) -> None:
        self._path(key).write_text(json.dumps(value))


@dataclass(frozen=True)
class ArmResult:
    urls: list[str]
    empty: bool
    error: str | None = None


@dataclass(frozen=True)
class QueryOutcome:
    domain: str
    query: str
    general: ArmResult
    domain_arm: ArmResult
    delta: float
    jaccard: float
    excluded: bool


async def _run_arm(query: str, *, search_fn, cache: DiskCache, provider: str) -> ArmResult:
    key = f"{provider}|{TOP_K}|{query}"
    cached = cache.get(key)
    if cached is not None:
        return ArmResult(urls=cached, empty=not cached)
    try:
        pages = await search_fn(query, provider=provider, page_size=TOP_K)
    except Exception as exc:  # provider error -> this arm is unusable
        return ArmResult(urls=[], empty=True, error=str(exc))
    urls = [p.url for p in pages if getattr(p, "url", None) and not getattr(p, "error", None)]
    cache.set(key, urls)
    return ArmResult(urls=urls, empty=not urls)


async def run_query(
    label: LabelledQuery, *, search_fn, cache: DiskCache, provider: str = "serpapi"
) -> QueryOutcome:
    """Run one labelled query through both arms and score them."""
    hinted = prepare_domain_query(label.query, label.domain)
    general = await _run_arm(label.query, search_fn=search_fn, cache=cache, provider=provider)
    domain_arm = await _run_arm(hinted, search_fn=search_fn, cache=cache, provider=provider)
    # A delta against a failed or empty arm measures that failure, not the
    # domain, so the query leaves the paired comparison.
    excluded = bool(general.error or domain_arm.error or general.empty or domain_arm.empty)
    delta = authority_precision(domain_arm.urls, label.authority_hosts) - authority_precision(
        general.urls, label.authority_hosts
    )
    return QueryOutcome(
        domain=label.domain,
        query=label.query,
        general=general,
        domain_arm=domain_arm,
        delta=delta,
        jaccard=jaccard(general.urls, domain_arm.urls),
        excluded=excluded,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_domain_eval.py -q`
Expected: all pass.

- [ ] **Step 5: Confirm no socket is opened**

Run: `pytest tests/unit/test_domain_eval.py -q -p no:cacheprovider`
Expected: passes in under two seconds, confirming nothing reached the network.

- [ ] **Step 6: Commit**

```bash
git add src/internal/retrieval/domain_eval.py tests/unit/test_domain_eval.py
git commit -m "feat(eval): add the cached two-arm runner for domain relevance"
```

---

### Task 4: CLI, statistics, and report

**Files:**
- Create: `examples/run_domain_relevance_eval.py`
- Test: `tests/unit/test_domain_eval.py`

**Interfaces:**
- Consumes: `load_labels`, `run_query`, `DiskCache`, `QueryOutcome` from Tasks 1-3; `paired_permutation_p` and `cliffs_delta` from `src.model.post_training.eval.stats`
- Produces: `summarise(outcomes: list[QueryOutcome]) -> dict` in the CLI module

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_domain_eval.py`:

```python
import importlib.util
import math

_SPEC = importlib.util.spec_from_file_location(
    "run_domain_relevance_eval",
    Path(__file__).resolve().parents[2] / "examples" / "run_domain_relevance_eval.py",
)


def _cli():
    module = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(module)
    return module


def _outcome(domain, delta, *, excluded=False, jaccard_value=0.5):
    from src.internal.retrieval.domain_eval import ArmResult, QueryOutcome

    arm = ArmResult(urls=["https://x.com/a"], empty=False)
    return QueryOutcome(
        domain=domain, query="q", general=arm, domain_arm=arm,
        delta=delta, jaccard=jaccard_value, excluded=excluded,
    )


def test_general_control_is_excluded_from_the_paired_deltas():
    # Three forced zeros would shrink the mean difference toward the null.
    summary = _cli().summarise(
        [_outcome("general", 0.0), _outcome("academic", 0.5), _outcome("code", 0.3)]
    )
    assert summary["paired"]["n"] == 2


def test_excluded_queries_do_not_enter_the_paired_deltas():
    summary = _cli().summarise(
        [_outcome("academic", 0.5), _outcome("code", 0.9, excluded=True)]
    )
    assert summary["paired"]["n"] == 1
    assert summary["excluded_count"] == 1


def test_no_per_domain_p_values_are_reported():
    # 17 tests at n=3 under BH correction could not reject anything; printing
    # them would present that vacuum as a finding.
    summary = _cli().summarise([_outcome("academic", 0.5), _outcome("code", 0.2)])
    for domain_stats in summary["per_domain"].values():
        assert "p_value" not in domain_stats


def test_every_reported_float_is_finite():
    # The Dev Console reads data/eval/*.json; a non-finite float arrives as null.
    summary = _cli().summarise([_outcome("academic", 0.5), _outcome("code", 0.2)])

    def _check(node):
        if isinstance(node, dict):
            for v in node.values():
                _check(v)
        elif isinstance(node, float):
            assert math.isfinite(node), node

    _check(summary)


def test_an_all_excluded_run_is_marked_uninterpretable():
    summary = _cli().summarise([_outcome("academic", 0.5, excluded=True)])
    assert summary["interpretable"] is False
    assert summary["paired"]["p_value"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_domain_eval.py -q -k "summar or control or per_domain or finite or uninterpretable"`
Expected: FAIL — the CLI module does not exist.

- [ ] **Step 3: Implement the CLI**

Create `examples/run_domain_relevance_eval.py`:

```python
"""Compare general and domain search arms on labelled queries.

Answers one question: does selecting a topic domain move results toward that
topic's authoritative sources? It measures source alignment, not answer
quality -- an arxiv.org result is not automatically a better answer.

The statistics live here rather than in
src/internal/retrieval/domain_eval.py because importing
src.model.post_training pulls torch in through package __init__ side
effects, and the measurement core must stay torch-free.

Usage:
    python -m examples.run_domain_relevance_eval --dry-run
    python -m examples.run_domain_relevance_eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from src.internal.retrieval.domain_eval import (
    DEFAULT_LABELS_PATH,
    DiskCache,
    QueryOutcome,
    load_labels,
    run_query,
)
from src.internal.tools.search import search_tool
from src.model.post_training.eval.stats import cliffs_delta, paired_permutation_p

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "eval" / "cache"
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "eval" / "domain_relevance.json"

# Below this share of usable queries the remainder is not worth reporting.
MIN_USABLE_SHARE = 2 / 3


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(outcomes: list[QueryOutcome]) -> dict:
    """Reduce per-query outcomes to the reported statistics.

    `general` is a control, not data: its hint is empty so its deltas are
    structurally zero, and including them would shrink the mean difference
    toward the null. Excluded queries leave for the same reason -- a delta
    against a failed arm measures the failure.
    """
    control = [o for o in outcomes if o.domain == "general"]
    topic = [o for o in outcomes if o.domain != "general"]
    usable = [o for o in topic if not o.excluded]
    deltas = [o.delta for o in usable]

    interpretable = bool(topic) and len(usable) >= MIN_USABLE_SHARE * len(topic)

    if deltas and interpretable:
        p_value = paired_permutation_p(deltas, alternative="two-sided")
        effect = cliffs_delta(deltas, [0.0] * len(deltas))
    else:
        p_value = None
        effect = None

    per_domain: dict[str, dict] = {}
    for outcome in topic:
        bucket = per_domain.setdefault(
            outcome.domain, {"deltas": [], "jaccards": [], "excluded": 0}
        )
        if outcome.excluded:
            bucket["excluded"] += 1
        else:
            bucket["deltas"].append(outcome.delta)
            bucket["jaccards"].append(outcome.jaccard)
    # Descriptive only: three queries cannot support a per-domain claim, so no
    # p-value is computed here at any point.
    per_domain = {
        name: {
            "mean_delta": _mean(b["deltas"]),
            "mean_jaccard": _mean(b["jaccards"]),
            "n_usable": len(b["deltas"]),
            "excluded": b["excluded"],
        }
        for name, b in per_domain.items()
    }

    return {
        "interpretable": interpretable,
        "excluded_count": sum(1 for o in topic if o.excluded),
        "empty_rate_general": _mean([float(o.general.empty) for o in topic]),
        "empty_rate_domain": _mean([float(o.domain_arm.empty) for o in topic]),
        "paired": {
            "n": len(deltas),
            "mean_delta": _mean(deltas),
            "p_value": p_value,
            "cliffs_delta": effect,
        },
        "control": {
            "n": len(control),
            "max_abs_delta": max((abs(o.delta) for o in control), default=0.0),
            "min_jaccard": min((o.jaccard for o in control), default=1.0),
        },
        "per_domain": per_domain,
    }


def _finite(node):
    """Replace non-finite floats with None before the report is written."""
    if isinstance(node, dict):
        return {k: _finite(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_finite(v) for v in node]
    if isinstance(node, float) and not math.isfinite(node):
        return None
    return node


async def _run(args) -> dict:
    labels = load_labels(args.labels)
    cache = DiskCache(args.cache_dir)
    outcomes = []
    for label in labels:
        outcomes.append(
            await run_query(label, search_fn=search_tool, cache=cache, provider=args.provider)
        )
        print(f"  {label.domain:<14} {label.query[:48]}")
    return summarise(outcomes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--provider", default="serpapi")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many live provider calls the run would make, then exit.",
    )
    args = parser.parse_args(argv)

    labels = load_labels(args.labels)
    cache = DiskCache(args.cache_dir)
    uncached = 0
    for label in labels:
        from src.internal.retrieval.domain_eval import TOP_K
        from src.internal.tools.search import prepare_domain_query

        for query in {label.query, prepare_domain_query(label.query, label.domain)}:
            if cache.get(f"{args.provider}|{TOP_K}|{query}") is None:
                uncached += 1
    print(f"{len(labels)} labelled queries; {uncached} live provider calls needed.")
    if args.dry_run:
        return

    summary = asyncio.run(_run(args))
    Path(args.report).write_text(json.dumps(_finite(summary), indent=2))
    print(json.dumps(_finite(summary)["paired"], indent=2))
    print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_domain_eval.py -q`
Expected: all pass.

- [ ] **Step 5: Verify the CLI runs without spending quota**

Run: `python -m examples.run_domain_relevance_eval --dry-run`
Expected: prints `48 labelled queries; 96 live provider calls needed.` and exits.
If the count is not 96, the cache key or the label set is wrong — fix before
spending any quota.

- [ ] **Step 6: Commit**

```bash
git add examples/run_domain_relevance_eval.py tests/unit/test_domain_eval.py
git commit -m "feat(eval): add the domain relevance CLI and paired statistics"
```

---

### Task 5: Run the evaluation and record the result

**Files:**
- Create: `data/eval/domain_relevance.json`
- Modify: `docs/search-engine.md`

- [ ] **Step 1: Check the remaining quota before spending it**

```bash
python3 -c "
import os, requests
from dotenv import load_dotenv; load_dotenv('.env')
r = requests.get('https://serpapi.com/account', params={'api_key': os.environ['SERP_API_KEY']}, timeout=20)
print(r.json().get('total_searches_left'), 'searches left')
"
```
Expected: a number comfortably above the count the dry run reported. If it is
not, stop and report rather than starting a run that will fail partway.

- [ ] **Step 2: Run the evaluation**

Run: `python -m examples.run_domain_relevance_eval`
Expected: writes `data/eval/domain_relevance.json` and prints the paired block.

- [ ] **Step 3: Check the control before reading any result**

Confirm `control.max_abs_delta` is 0.0 and `control.min_jaccard` is 1.0 in the
report. A non-zero delta or a jaccard below 1.0 on `general` means the harness
is not pairing arms correctly, and the run must be rejected rather than
reported.

- [ ] **Step 4: Document the evaluation**

Add a subsection to `docs/search-engine.md` under "Search topic domains"
covering: how to run the evaluation, what authority precision measures and what
it does not, that the unit of analysis is the query, that per-domain figures
are descriptive only, the observed pooled result stated plainly including if it
is null, and that a null at this sample size means no effect was detected
rather than that none exists.

- [ ] **Step 5: Commit, force-adding the report**

```bash
git add -f data/eval/domain_relevance.json
git add docs/search-engine.md
git commit -m "eval: record the domain relevance result"
```

---

## Self-Review

**Spec coverage:** environment constraints drove the design and need no task;
source-authority relevance (Tasks 1-2); the 16-topic/`general`-control split
(Tasks 2-4); authority precision, empty rate, jaccard (Tasks 1, 3, 4); k=10
(Task 1 `TOP_K`); query as the unit of analysis and the pooled paired test
(Task 4); per-domain descriptive only (Task 4, asserted by test); control
excluded from deltas (Task 4, asserted by test); cache and cost (Task 3, dry
run in Task 4); exclusion on failed or empty arms and the two-thirds
interpretability threshold (Tasks 3-4); loud failure on a bad label file (Task
2); finite floats (Task 4); the live run and its honest write-up (Task 5).

**Placeholder scan:** no TBD or "handle errors appropriately" steps; every code
step carries runnable code.

**Type consistency:** `LabelledQuery(domain, query, authority_hosts)`,
`ArmResult(urls, empty, error)`, and `QueryOutcome(domain, query, general,
domain_arm, delta, jaccard, excluded)` are used with identical field names in
Tasks 2, 3, and 4. `TOP_K` is defined once in Task 1 and reused in the cache
key in Tasks 3 and 4. `DEFAULT_LABELS_PATH` is defined in Task 2 and imported
in Task 4.
