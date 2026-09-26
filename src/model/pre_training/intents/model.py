"""The intent routing model: its taxonomy, its encoder, and its index.

Routing is nearest-neighbour over curated canonical examples. The route is the
argmax of a per-route score: the mean of the top-k cosine similarities among
that route's canonical examples, ``TOP_K`` by default. ``top_k`` is a
parameter, not a hardcoded constant, and is selected on the tuning slice (see
docs/training-and-evaluation.md).

**One** threshold gates the result: a low margin between the best and
second-best route means two routes fit equally well, so the request is
ambiguous and the caller falls through to the LLM classifier. There used to be
a second, absolute-confidence gate for out-of-scope requests; it was removed
after measurement showed it changed 3 decisions out of 416 — see ``decide``.

Cosine is deliberately not normalized across routes. A softmax head sums to one
by construction and so cannot express "none of these", which is why the previous
model's out-of-scope separation was only +0.059.

The three sections below were three modules until they were merged here. The
layering they encoded still holds and still matters:

* **Taxonomy** depends on nothing. It is the vocabulary everything else speaks.
* **Encoder** is the only place a sentence encoder is loaded, and its
  ``sentence_transformers`` import stays function-local. Every other part of the
  routing path must stay importable in a CI job that installs neither torch nor
  sentence-transformers, and this repo has twice shipped collection failures
  from unguarded imports (#356, re-fixed in #418).
* **Index** is the scoring itself, and imports numpy and nothing heavier.

Keeping the encoder dependency behind one function is what makes the rest of
the routing path testable without it.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.shared_configs.intent import INTENT_LABELS
from src.shared_configs.intent import DEFAULT_TOP_K as TOP_K

# ---------------------------------------------------------------------------
# Taxonomy — the two-level intent taxonomy: three routes, each with its own
# modules.
#
# Modules are route-scoped rather than orthogonal primitives. At a few hundred
# canonical examples a small hierarchical taxonomy is labelable consistently and
# explainable; orthogonal primitives are the right end state at several thousand.
#
# The module names are not invented. Each is drawn from a regex cue already used
# by ``src/internal/servers/web/intent/rules.py``, so the taxonomy describes
# distinctions the router already makes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSpec:
    """One module: its route, and whether it names an intent or an utterance form."""

    name: str
    route: str
    kind: str


_SPECS: tuple[ModuleSpec, ...] = (
    # search — _SEARCH_LOOKUP_RE, _CURRENCY_RE, _is_bare_lookup
    ModuleSpec("lookup_document", "search", "intent"),
    ModuleSpec("lookup_fact", "search", "intent"),
    ModuleSpec("current_info", "search", "intent"),
    # A form label, not an intent: "OpenAI" is a bare entity, "OpenAI CEO" is a
    # fact lookup. It is excluded from module macro-F1, and _is_bare_lookup
    # routes such queries at cascade step 2, before this model ever runs.
    ModuleSpec("bare_entity", "search", "form"),
    # chat — _CHAT_START_RE, _GENERATIVE_RE
    ModuleSpec("explain", "chat", "intent"),
    ModuleSpec("summarize", "chat", "intent"),
    ModuleSpec("compare", "chat", "intent"),
    ModuleSpec("generate", "chat", "intent"),
    ModuleSpec("converse", "chat", "intent"),
    # tool — _TOOL_ACTION_RE, _TOOL_OBJECT_RE
    ModuleSpec("create", "tool", "intent"),
    ModuleSpec("send", "tool", "intent"),
    ModuleSpec("schedule", "tool", "intent"),
    ModuleSpec("modify", "tool", "intent"),
    ModuleSpec("execute", "tool", "intent"),
)

MODULES: dict[str, ModuleSpec] = {spec.name: spec for spec in _SPECS}

SEMANTIC_MODULES: tuple[str, ...] = tuple(
    spec.name for spec in _SPECS if spec.kind == "intent"
)

# Composite detection keys off these: a runner-up route whose best module is an
# action is the signature of "find X and book it".
ACTION_MODULES: frozenset[str] = frozenset(
    spec.name for spec in _SPECS if spec.route == "tool"
)


def modules_for_route(route: str) -> tuple[str, ...]:
    """Every module belonging to *route*, in taxonomy order."""
    return tuple(spec.name for spec in _SPECS if spec.route == route)


def route_of_module(module: str) -> str:
    """The route *module* belongs to. Raises KeyError if unknown."""
    return MODULES[module].route


def validate_modules(route: str, modules: Sequence[str]) -> None:
    """Check a label's modules: nonempty, known, unique, and all in *route*."""
    if not modules:
        raise ValueError(f"Route {route!r} needs at least one module")
    seen: set[str] = set()
    for module in modules:
        if module in seen:
            raise ValueError(f"Found duplicate module {module!r} for route {route!r}")
        seen.add(module)
        spec = MODULES.get(module)
        if spec is None:
            raise ValueError(f"Unknown module {module!r}")
        if spec.route != route:
            raise ValueError(
                f"Module {module!r} belongs to route {spec.route!r}, not {route!r}"
            )


# ---------------------------------------------------------------------------
# Encoder — the only place a sentence encoder is loaded. The
# ``sentence_transformers`` import below is function-local on purpose; see the
# module docstring.
# ---------------------------------------------------------------------------
DEFAULT_ENCODER = "intfloat/e5-small-v2"

# E5 models are trained with instruction prefixes and degrade *silently*
# without them — no error, just worse vectors. The prefix is therefore a
# property of the model, derived from its name rather than passed by callers,
# so no call site can omit it. Deriving rather than storing it also means the
# index needs no new field: it already records the encoder name, and
# intent.similarity.load_intent_index already rejects a mismatch, which covers the
# prefix for free. That matters here because e5-small-v2 is also 384-wide, so
# an index built with the previous encoder would otherwise load and score
# without any error at all.
#
# Both sides of the comparison use "query: ": this is symmetric short-text
# similarity, not the asymmetric query/passage retrieval "passage: " is for.
MODEL_PREFIXES: dict[str, str] = {
    "intfloat/e5-small-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
}


def prefix_for(model_name: str) -> str:
    """The instruction prefix *model_name* requires.

    Raises rather than defaulting to "": an unregistered model is far more
    likely to be one whose prefix nobody looked up than one that genuinely
    needs none, and guessing wrong is invisible.
    """
    try:
        return MODEL_PREFIXES[model_name]
    except KeyError:
        raise ValueError(
            f"No instruction prefix registered for encoder {model_name!r}. "
            f"Add it to MODEL_PREFIXES — encoders that need a prefix degrade "
            f"silently without one. Known: {sorted(MODEL_PREFIXES)}"
        ) from None


# Keyed by model name; holds either a loaded model or an Exception. A failed
# load is cached too — lru_cache does not cache exceptions, and retrying a
# broken or unreachable model download on every call would block whichever
# request triggers it, forever. See intent.similarity._INTENT_INDEXES for the same
# policy one layer up.
_MODEL_CACHE: dict[str, object] = {}
# Serving encodes from worker threads (#657): one cold load per encoder name.
_MODEL_LOCK = threading.Lock()


def _model(model_name: str):
    """Load and cache the encoder. Loading costs seconds; encoding costs ms."""
    cached = _MODEL_CACHE.get(model_name)
    if cached is None:
        with _MODEL_LOCK:
            cached = _MODEL_CACHE.get(model_name)
            if cached is None:
                from sentence_transformers import SentenceTransformer

                try:
                    model = SentenceTransformer(model_name, device="cpu")
                except Exception as exc:
                    _MODEL_CACHE[model_name] = exc
                    raise
                _MODEL_CACHE[model_name] = model
                return model
    if isinstance(cached, Exception):
        raise RuntimeError(
            f"intent encoder {model_name!r} failed to load previously; not retrying"
        ) from cached
    return cached


def encode_texts(
    texts: Sequence[str], *, model_name: str = DEFAULT_ENCODER
) -> np.ndarray:
    """Encode *texts* to L2-normalized float32 rows.

    Normalizing here means every consumer can treat a dot product as a cosine,
    and the index constructor can reject anything that is not normalized.
    """
    prefix = prefix_for(model_name)
    vectors = _model(model_name).encode(
        [prefix + text for text in texts],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Index — canonical example vectors and the scoring rules over them.
# ---------------------------------------------------------------------------
MIN_MODULE_SUPPORT = 10
INDEX_FILENAME = "index.npz"

_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class CanonicalExample:
    """One curated routing example: the unit of the index."""

    id: str
    text: str
    route: str
    modules: tuple[str, ...]


@dataclass(frozen=True)
class KnnDecision:
    """A route decision with its diagnostics and, if any, its abstention."""

    route: str
    confidence: float
    margin: float
    modules: tuple[str, ...]
    composite: bool
    abstained: bool
    abstain_reason: str | None


class IntentIndex:
    """Canonical example vectors and the scoring rules over them."""

    def __init__(
        self,
        examples: Sequence[CanonicalExample],
        vectors: np.ndarray,
        encoder: str,
        fingerprint: str,
    ) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(examples):
            raise ValueError(
                "Index rows must equal the example count: "
                f"{vectors.shape} against {len(examples)} examples"
            )
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
            raise ValueError(
                "Index vectors must be L2-normalized; cosine similarity is "
                "otherwise an arbitrary dot product with no later symptom"
            )
        for example in examples:
            validate_modules(example.route, example.modules)

        self._examples = tuple(examples)
        self._vectors = vectors.astype(np.float32)
        self._encoder = encoder
        self._fingerprint = fingerprint
        self._route_rows = {
            route: np.array(
                [i for i, e in enumerate(self._examples) if e.route == route],
                dtype=np.int64,
            )
            for route in INTENT_LABELS
        }
        self._module_rows = {
            module: np.array(
                [i for i, e in enumerate(self._examples) if module in e.modules],
                dtype=np.int64,
            )
            for module in (
                m for route in INTENT_LABELS for m in modules_for_route(route)
            )
        }

    @property
    def size(self) -> int:
        return len(self._examples)

    @property
    def encoder(self) -> str:
        return self._encoder

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def examples(self) -> tuple[CanonicalExample, ...]:
        return self._examples

    @property
    def vectors(self) -> np.ndarray:
        """The normalized example matrix, for callers that need raw similarity."""
        return self._vectors

    def _top_k_mean(
        self, similarities: np.ndarray, rows: np.ndarray, top_k: int
    ) -> float:
        """Mean of the highest *top_k* similarities among *rows*, or 0.0 if empty.

        Taking a top-k mean rather than the mean of all rows keeps one close
        neighbor from being diluted by distant same-route examples, and keeps
        one outlier from carrying the route the way a bare max would.
        """
        if rows.size == 0:
            return 0.0
        selected = similarities[rows]
        if selected.size > top_k:
            selected = np.partition(selected, -top_k)[-top_k:]
        return float(selected.mean())

    def _similarities(self, vector: np.ndarray) -> np.ndarray:
        query = np.asarray(vector, dtype=np.float32)
        expected = (self._vectors.shape[1],)
        if query.shape != expected:
            # einsum's own error for a mismatched operand is about subscripts,
            # not about the caller's mistake; say what was expected.
            raise ValueError(
                f"query vector has shape {query.shape}; the index expects a "
                f"single vector of shape {expected}"
            )
        # This is a small matrix-vector reduction, not a batched BLAS workload.
        # BLAS workers can keep spinning after scoring and contend with the
        # next CPU encoder call. Keep the contraction on the calling thread;
        # optimize=True may dispatch it back through BLAS.
        return np.einsum("ij,j->i", self._vectors, query, optimize=False)

    def route_scores(
        self, vector: np.ndarray, *, top_k: int = TOP_K
    ) -> dict[str, float]:
        """Per-route top-k mean cosine."""
        similarities = self._similarities(vector)
        return {
            route: self._top_k_mean(similarities, rows, top_k)
            for route, rows in self._route_rows.items()
        }

    def module_scores(
        self, vector: np.ndarray, *, top_k: int = TOP_K
    ) -> dict[str, float]:
        """Per-module top-k mean cosine, over every module in the taxonomy."""
        similarities = self._similarities(vector)
        return {
            module: self._top_k_mean(similarities, rows, top_k)
            for module, rows in self._module_rows.items()
        }

    def low_support_modules(self, minimum: int = MIN_MODULE_SUPPORT) -> tuple[str, ...]:
        """Modules with too few examples to score meaningfully."""
        return tuple(
            module for module, rows in self._module_rows.items() if rows.size < minimum
        )

    def decide(
        self,
        vector: np.ndarray,
        *,
        min_margin: float,
        min_module_score: float,
        top_k: int = TOP_K,
    ) -> KnnDecision:
        """Score *vector*, apply the margin threshold, and report modules.

        **There was a second gate here** -- an absolute ``min_confidence``
        floor, meant to catch requests that resemble nothing canonical. It was
        removed after being swept on the tuning slice: the rule selected a value
        below the lowest in-scope score, and across every evaluation set
        available (416 decisions) the floor changed **3** of them, two of which
        were the tuning probes it had been selected from.

        The reason it earned nothing is that the two gates overlap. Under this
        encoder in-scope and out-of-scope scores occupy the same narrow band, so
        anything far enough from one route to fail an absolute floor is already
        close to two routes and fails the margin. Keeping a knob that looks like
        out-of-scope protection but never fires is worse than not having it,
        because it invites tuning that does nothing and hides the absence of the
        control it appears to offer.

        A future encoder that separates absolute scores cleanly would need it
        back; the git history has it.
        """
        routes = self.route_scores(vector, top_k=top_k)
        ranked = sorted(routes.items(), key=lambda item: item[1], reverse=True)
        (best_route, confidence), (runner_up, runner_up_score) = ranked[0], ranked[1]
        margin = confidence - runner_up_score

        abstain_reason = "margin_below_threshold" if margin < min_margin else None

        modules = self._emit_modules(vector, best_route, min_module_score, top_k=top_k)
        return KnnDecision(
            route=best_route,
            confidence=confidence,
            margin=margin,
            modules=modules,
            composite=self._is_composite(
                vector, runner_up, margin, min_margin, top_k=top_k
            ),
            abstained=abstain_reason is not None,
            abstain_reason=abstain_reason,
        )

    def _emit_modules(
        self,
        vector: np.ndarray,
        route: str,
        min_module_score: float,
        *,
        top_k: int = TOP_K,
    ) -> tuple[str, ...]:
        """Every well-supported module of *route* scoring at or above the bar.

        Falls back to the single best so a decision always carries a module.
        This is multi-label because real requests carry several intents at once:
        "compare the current prices of BTC and ETH" is both current_info and
        lookup_fact.
        """
        low_support = set(self.low_support_modules())
        scores = self.module_scores(vector, top_k=top_k)
        # A module with zero examples has no real score at all (module_scores
        # reports 0.0 for it by construction) and must never be a candidate,
        # even as a fallback. "Low support" (< MIN_MODULE_SUPPORT) is a softer,
        # ignorable-under-fallback signal for modules that do have examples.
        populated = {
            module: scores[module]
            for module in modules_for_route(route)
            if self._module_rows[module].size > 0
        }
        candidates = {
            module: score
            for module, score in populated.items()
            if module not in low_support
        }
        if not candidates:
            candidates = populated
        emitted = tuple(
            module for module, score in candidates.items() if score >= min_module_score
        )
        if emitted:
            return emitted
        return (max(candidates, key=lambda module: candidates[module]),)

    def _is_composite(
        self,
        vector: np.ndarray,
        runner_up: str,
        margin: float,
        min_margin: float,
        *,
        top_k: int = TOP_K,
    ) -> bool:
        """True when a close runner-up route is an action.

        This is the signature of a request that needs two steps — "find the best
        Italian place near the office and book it for 7". Nothing acts on the
        flag yet; it is recorded so the plan-aware router can be designed
        against measured data rather than guesses.
        """
        if margin >= min_margin:
            return False
        scores = self.module_scores(vector, top_k=top_k)
        candidates = modules_for_route(runner_up)
        if not candidates:
            return False
        best = max(candidates, key=lambda module: scores[module])
        return best in ACTION_MODULES

    def save(self, path: Path) -> None:
        """Write vectors and labels to a single npz."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=self._vectors,
            examples=np.array(
                json.dumps(
                    [
                        {
                            "id": e.id,
                            "text": e.text,
                            "route": e.route,
                            "modules": list(e.modules),
                        }
                        for e in self._examples
                    ]
                )
            ),
            encoder=np.array(self._encoder),
            fingerprint=np.array(self._fingerprint),
        )

    @classmethod
    def load(cls, path: Path) -> "IntentIndex":
        """Read an index written by ``save``."""
        if not path.exists():
            raise FileNotFoundError(
                f"Intent index is missing {path.name}: {path}. Run "
                "`python -m src.model.pre_training.intents.cli build` to create it."
            )
        payload = np.load(path, allow_pickle=False)
        records = json.loads(str(payload["examples"]))
        examples = [
            CanonicalExample(
                id=record["id"],
                text=record["text"],
                route=record["route"],
                modules=tuple(record["modules"]),
            )
            for record in records
        ]
        return cls(
            examples=examples,
            vectors=payload["vectors"],
            encoder=str(payload["encoder"]),
            fingerprint=str(payload["fingerprint"]),
        )
