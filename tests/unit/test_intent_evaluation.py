import functools
import json
from pathlib import Path

import numpy as np
import pytest

from src.model.pre_training.intents import data as intent_index_data
from src.model.pre_training.intents import evaluation as intent_index_eval
from src.model.pre_training.intents.model import CanonicalExample, IntentIndex

# Orthogonal one-hot vectors in (search, chat, tool) component order, so
# top-3-mean cosine similarity is exactly 0.0 or 1.0 for a clean match and any
# fractional value we choose for a deliberately ambiguous query.
_SEARCH = [1.0, 0.0, 0.0]
_CHAT = [0.0, 1.0, 0.0]
_TOOL = [0.0, 0.0, 1.0]

_MODULE_FOR_ROUTE = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}

_CANONICAL = [
    {
        "id": f"c-{route}-{i}",
        "text": f"canonical {route} {i}",
        "route": route,
        "modules": [_MODULE_FOR_ROUTE[route]],
    }
    for route in ("search", "chat", "tool")
    for i in range(2)
]

_VECTORS_BY_TEXT = {
    **{
        record["text"]: vector
        for record, vector in zip(
            _CANONICAL, [_SEARCH, _SEARCH, _CHAT, _CHAT, _TOOL, _TOOL]
        )
    },
    # 0.85 on-axis, not 1.0: a clear, correct match without tripping the
    # >=0.95 leakage guard the way an exact-axis vector would.
    "legacy search query": [0.85, 0.05, 0.05],
    "legacy chat query": [0.05, 0.85, 0.05],
    "clean search query": [0.85, 0.05, 0.05],
    "clean tool query": [0.05, 0.05, 0.85],
    # Deliberately mislabeled ("tool") and low confidence (best route scores
    # only 0.3): the worst case for a threshold-gated accuracy computation,
    # since a buggy filter would most plausibly drop exactly this kind of
    # record and inflate accuracy by omission.
    "clean low confidence miss": [0.3, 0.25, 0.2],
    "probe one": [0.1, 0.05, 0.02],
    "probe two": [0.2, 0.0, 0.0],
}


def _encode_stub(texts, *, model_name="test-encoder"):
    return np.array([_VECTORS_BY_TEXT[text] for text in texts], dtype=np.float32)


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build_index(tmp_path: Path) -> Path:
    output = tmp_path / "index"
    intent_index_data.build_index(
        _write_json(tmp_path / "canonical.json", _CANONICAL),
        output,
        model_name="test-encoder",
        encode=_encode_stub,
    )
    return output


def _eval_queries_path(tmp_path: Path, records) -> Path:
    return _write_json(tmp_path / "eval_queries.json", records)


_BULK_QUERIES = [
    {
        "id": "eval-a",
        "text": "legacy search query",
        "label": "search",
        "modules": ["lookup_fact"],
    },
    {
        "id": "eval-b",
        "text": "legacy chat query",
        "label": "chat",
        "modules": ["explain"],
    },
    {
        "id": "bulk-a",
        "text": "clean search query",
        "label": "search",
        "modules": ["lookup_fact"],
    },
    {
        "id": "bulk-b",
        "text": "clean tool query",
        "label": "tool",
        "modules": ["schedule"],
    },
    {
        "id": "bulk-c",
        "text": "clean low confidence miss",
        "label": "tool",
        "modules": ["schedule"],
    },
]

_PROBES = [
    {"id": "oos-1", "text": "probe one"},
    {"id": "oos-2", "text": "probe two"},
]

# Reuses texts already in _VECTORS_BY_TEXT / _BULK_QUERIES, so no new encoder
# stub entries are needed and the 0.85-on-axis vectors stay well clear of the
# >=0.95 leakage guard.
_HARD_QUERIES = [
    {
        "id": "hard-a",
        "text": "clean search query",
        "label": "search",
        "modules": ["lookup_fact"],
    },
    {
        "id": "hard-b",
        "text": "clean tool query",
        "label": "tool",
        "modules": ["schedule"],
    },
]


# Only 3 clean (bulk-) queries exist in this fixture, so slice_size must stay
# well under that. At 1, the split is deterministic regardless of seed: the
# lone "search"-route clean query is the only member of its pool, so
# rng.sample of a 1-item list always returns that same item.
_SLICE_SIZE = 1


def _run(tmp_path, *, monkeypatch, out_of_scope=None, hard=None):
    monkeypatch.setattr(intent_index_eval, "encode_texts", _encode_stub)
    index_dir = _build_index(tmp_path)
    return intent_index_eval.run_index_evaluation(
        index_path=index_dir,
        eval_queries_path=_eval_queries_path(tmp_path, _BULK_QUERIES),
        hard_queries_path=(
            _write_json(tmp_path / "hard.json", _HARD_QUERIES) if hard else None
        ),
        out_of_scope_path=(
            _write_json(tmp_path / "oos.json", _PROBES) if out_of_scope else None
        ),
        canonical_path=tmp_path / "canonical.json",
        output_path=tmp_path / "report.json",
        model_name="test-encoder",
        slice_size=_SLICE_SIZE,
    )


def test_split_puts_every_legacy_query_in_tuning_and_reports_split_sizes(
    tmp_path, monkeypatch
):
    """The legacy queries are contaminated, so all of them are free to tune on.

    With ``_SLICE_SIZE`` clean queries drawn into tuning, tuning holds the 2
    legacy queries plus that slice, and the rest of the clean set becomes the
    untouched test slice.
    """
    report = _run(tmp_path, monkeypatch=monkeypatch)

    assert report["split"]["tuning_size"] == 2 + _SLICE_SIZE
    assert report["split"]["test_size"] == 3 - _SLICE_SIZE
    assert report["legacy_30"]["total_queries"] == 2
    assert report["tuning"]["total_queries"] == report["split"]["tuning_size"]
    assert report["test_slice"]["total_queries"] == report["split"]["test_size"]
    assert report["bulk"]["total_queries"] == len(_BULK_QUERIES)
    # Pooled, not just parallel: nothing dropped, nothing double-counted.
    assert (
        report["tuning"]["total_queries"] + report["test_slice"]["total_queries"]
        == report["bulk"]["total_queries"]
    )


def test_accuracy_is_argmax_over_every_query_not_only_those_above_a_threshold(
    tmp_path, monkeypatch
):
    """A confidently-wrong, low-similarity prediction must still count.

    If accuracy were computed only over records clearing some confidence
    threshold, the low-confidence miss ("clean low confidence miss", best
    route score only 0.3) is exactly the record a bug would drop. With
    ``_SLICE_SIZE`` == 1 the split is deterministic: the lone search-route
    clean query ("clean search query") is the only one that can join tuning,
    so the test slice always holds the other two -- the correct tool query
    and the mislabeled low-confidence miss.
    """
    report = _run(tmp_path, monkeypatch=monkeypatch)

    assert report["legacy_30"]["accuracy"] == pytest.approx(1.0)
    assert report["tuning"]["accuracy"] == pytest.approx(1.0)
    assert report["test_slice"]["accuracy"] == pytest.approx(1 / 2)
    assert report["bulk"]["accuracy"] == pytest.approx(4 / 5)


def test_out_of_scope_separability_is_reported_on_held_out_probes_only(
    tmp_path, monkeypatch
):
    """The reported AUC must not be measured on probes that tuned the sweep.

    Until the probe set grew large enough to split, one set of probes both
    tie-broke threshold selection and denominated the reported separability --
    so the headline figure was never fully held out from the thresholds it was
    measured at. The report now carries `probe_split`, and `out_of_scope`
    counts only the reporting half.
    """
    report = _run(tmp_path, monkeypatch=monkeypatch, out_of_scope=True)

    split = report["probe_split"]
    assert split["tuning"] >= 1 and split["reporting"] >= 1
    assert split["tuning"] + split["reporting"] == len(_PROBES)

    out_of_scope = report["out_of_scope"]
    assert out_of_scope["tuned_on"] is False
    assert out_of_scope["probes"] == split["reporting"]
    assert out_of_scope["counts"]["out_of_scope"] == split["reporting"]
    # In-scope side is the whole test slice, unaffected by the probe split.
    assert out_of_scope["counts"]["in_scope"] == report["split"]["test_size"]
    # Still perfectly separated on this fixture: every in-scope score (0.85,
    # 0.3) clears every probe score (0.1, 0.2), whichever half each landed in.
    assert out_of_scope["auc"] == pytest.approx(1.0)


def test_the_probe_split_halves_are_disjoint_and_cover_every_probe(tmp_path):
    """A probe leaking into both halves would silently undo the split."""
    from src.model.pre_training.intents.evaluation import split_out_of_scope_probes

    probes = tuple((p["id"], p["text"]) for p in _PROBES)
    split = split_out_of_scope_probes(probes)

    tuning_ids = {pid for pid, _ in split.tuning}
    reporting_ids = {pid for pid, _ in split.reporting}
    assert not (tuning_ids & reporting_ids)
    assert tuning_ids | reporting_ids == {pid for pid, _ in probes}
    # Deterministic in the seed alone.
    assert split_out_of_scope_probes(probes) == split
    assert split_out_of_scope_probes(tuple(reversed(probes))) == split


def test_run_index_evaluation_raises_on_a_stale_canonical_fingerprint(
    tmp_path, monkeypatch
):
    """Editing the canonical file without rebuilding must fail loudly.

    Otherwise `evaluate` silently scores the index built from the *old*
    canonical file and stamps the report with the new canonical path, so an
    operator reads the old accuracy as if it measured their edit.
    """
    monkeypatch.setattr(intent_index_eval, "encode_texts", _encode_stub)
    index_dir = _build_index(tmp_path)
    canonical_path = tmp_path / "canonical.json"
    edited = json.loads(canonical_path.read_text(encoding="utf-8"))
    edited[0]["text"] = "canonical search 0 but edited after the index was built"
    canonical_path.write_text(json.dumps(edited), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        intent_index_eval.run_index_evaluation(
            index_path=index_dir,
            eval_queries_path=_eval_queries_path(tmp_path, _BULK_QUERIES),
            hard_queries_path=None,
            out_of_scope_path=None,
            canonical_path=canonical_path,
            output_path=tmp_path / "report.json",
            model_name="test-encoder",
            slice_size=_SLICE_SIZE,
        )


def test_run_index_evaluation_passes_when_the_canonical_fingerprint_matches(
    tmp_path, monkeypatch
):
    report = _run(tmp_path, monkeypatch=monkeypatch)

    assert report["index"]["fingerprint"]


def test_run_index_evaluation_raises_on_leakage_instead_of_scoring_anyway(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(intent_index_eval, "encode_texts", _encode_stub)
    index_dir = _build_index(tmp_path)
    leaking_queries = [
        {
            "id": "bulk-leak",
            "text": "canonical search 0",
            "label": "search",
            "modules": ["lookup_fact"],
        },
    ]

    with pytest.raises(ValueError, match="leak"):
        intent_index_eval.run_index_evaluation(
            index_path=index_dir,
            eval_queries_path=_eval_queries_path(tmp_path, leaking_queries),
            hard_queries_path=None,
            out_of_scope_path=None,
            canonical_path=tmp_path / "canonical.json",
            output_path=tmp_path / "report.json",
            model_name="test-encoder",
            slice_size=1,
        )


def test_run_index_evaluation_raises_when_the_encoder_does_not_match_the_index(
    tmp_path, monkeypatch
):
    """Both all-MiniLM-L6-v2 and e5-small-v2 are 384-dimensional, so nothing
    else catches a mismatched encoder: no shape error, no exception, just a
    confident, meaningless number -- exactly what happened once already,
    scoring a stale MiniLM-built index against e5-encoded queries.

    The index built here is structurally valid (right dimensionality, right
    row count); its only defect is that it declares a different encoder than
    the one this evaluation asks for. ``_encode_stub`` ignores ``model_name``
    entirely, so without the guard this would run to completion and score
    without error -- the guard is the only thing standing between a real
    mismatch and a silently wrong report.
    """
    monkeypatch.setattr(intent_index_eval, "encode_texts", _encode_stub)
    index_dir = _build_index(tmp_path)  # built with model_name="test-encoder"

    with pytest.raises(ValueError) as exc_info:
        intent_index_eval.run_index_evaluation(
            index_path=index_dir,
            eval_queries_path=_eval_queries_path(tmp_path, _BULK_QUERIES),
            hard_queries_path=None,
            out_of_scope_path=None,
            canonical_path=tmp_path / "canonical.json",
            output_path=tmp_path / "report.json",
            model_name="different-encoder",
            slice_size=_SLICE_SIZE,
        )

    message = str(exc_info.value)
    assert "test-encoder" in message
    assert "different-encoder" in message


def test_leave_one_out_excludes_each_example_from_its_own_scoring():
    """A singleton route's only example cannot vote for itself once excluded.

    "tool" has exactly one canonical example. Scored against the full index
    including itself (the "control" a self-inclusive scorer would produce),
    it always wins its own route with similarity 1.0. Excluded from its own
    route's candidate pool (real leave-one-out), that route has no
    remaining examples at all, so the example is misclassified. The control
    must therefore score strictly higher than genuine leave-one-out on this
    fixture.
    """
    examples = [
        CanonicalExample("s0", "search zero", "search", ("lookup_fact",)),
        CanonicalExample("s1", "search one", "search", ("lookup_fact",)),
        CanonicalExample("c0", "chat zero", "chat", ("explain",)),
        CanonicalExample("c1", "chat one", "chat", ("explain",)),
        CanonicalExample("t0", "tool zero", "tool", ("schedule",)),
    ]
    vectors = np.array([_SEARCH, _SEARCH, _CHAT, _CHAT, _TOOL], dtype=np.float32)
    index = IntentIndex(
        examples=examples, vectors=vectors, encoder="test", fingerprint="test"
    )

    control_correct = sum(
        index.decide(vectors[i], min_margin=0.0, min_module_score=0.0).route
        == examples[i].route
        for i in range(len(examples))
    )
    control_accuracy = control_correct / len(examples)

    loo = intent_index_eval.leave_one_out_route_accuracy(index)

    assert control_accuracy == pytest.approx(1.0)
    assert loo["n"] == len(examples)
    assert loo["accuracy"] < control_accuracy


def test_leave_one_out_default_top_k_matches_the_module_constant():
    """Nothing passing top_k must behave exactly as before the parameter existed."""
    examples = [
        CanonicalExample("s0", "search zero", "search", ("lookup_fact",)),
        CanonicalExample("s1", "search one", "search", ("lookup_fact",)),
        CanonicalExample("c0", "chat zero", "chat", ("explain",)),
        CanonicalExample("c1", "chat one", "chat", ("explain",)),
        CanonicalExample("t0", "tool zero", "tool", ("schedule",)),
    ]
    vectors = np.array([_SEARCH, _SEARCH, _CHAT, _CHAT, _TOOL], dtype=np.float32)
    index = IntentIndex(
        examples=examples, vectors=vectors, encoder="test", fingerprint="test"
    )

    default = intent_index_eval.leave_one_out_route_accuracy(index)
    explicit = intent_index_eval.leave_one_out_route_accuracy(
        index, top_k=intent_index_eval.TOP_K
    )

    assert default == explicit


def test_top_k_sweep_reports_every_configured_k_without_changing_the_shipped_report(
    tmp_path, monkeypatch
):
    """The sweep is evidence, not a selection: the shipped headline is untouched.

    It also must never publish a fitting curve over held-out data -- the
    per-k accuracy and separation-margin columns are computed on *tuning*
    (legacy_a/legacy_b/bulk-a, all correctly and confidently routed at 0.85),
    never on test_slice (bulk-b/bulk-c) and never on hard-40, which is
    held-out test data by this project's own split. ``hard=True`` here
    supplies hard queries -- the shipping configuration, since the documented
    CLI always passes ``--hard-queries`` -- so this is the one configuration
    that actually exercises the guard.
    """
    report = _run(tmp_path, monkeypatch=monkeypatch, out_of_scope=True, hard=True)

    sweep = report["top_k_sweep"]
    assert [row["top_k"] for row in sweep["rows"]] == list(
        intent_index_eval._SWEEP_TOP_K
    )
    for row in sweep["rows"]:
        assert 0.0 <= row["tuning_accuracy"] <= 1.0
        assert 0.0 <= row["leave_one_out_accuracy"] <= 1.0
        assert "separation_margin" in row
        assert "hard_accuracy" not in row  # hard-40 is held-out; never in the sweep

    # The row at the shipped TOP_K reproduces the report's own tuning numbers.
    shipped = next(
        row for row in sweep["rows"] if row["top_k"] == intent_index_eval.TOP_K
    )
    assert shipped["tuning_accuracy"] == pytest.approx(report["tuning"]["accuracy"])
    assert shipped["leave_one_out_accuracy"] == pytest.approx(
        report["leave_one_out"]["accuracy"]
    )
    # tuning in-scope confidences are all 0.85 (legacy_a, legacy_b, bulk-a).
    # The probe side is the TUNING half only, since this sweep is a tuning
    # artifact -- derived here rather than hardcoded so the assertion tracks the
    # split instead of a value that changes whenever the split does.
    from src.model.pre_training.intents.evaluation import split_out_of_scope_probes

    tuning_probes = split_out_of_scope_probes(
        tuple((probe["id"], probe["text"]) for probe in _PROBES)
    ).tuning
    expected_probe_mean = sum(
        _VECTORS_BY_TEXT[text][0] for _, text in tuning_probes
    ) / len(tuning_probes)
    # Deliberately *not* compared against report["out_of_scope"]
    # ["separation_margin"]: that measures the test slice against the
    # *reporting* probes, two different slices by design.
    assert shipped["separation_margin"] == pytest.approx(0.85 - expected_probe_mean)


def test_the_sweep_searches_top_k_only_over_the_pre_registered_grid(
    tmp_path, monkeypatch
):
    """Replaces ``test_the_threshold_sweep_never_chooses_top_k``.

    That test guarded a property this repo deliberately gave up in #522: with
    ``k`` pinned, no threshold the sweep chose could move the abstention-blind
    argmax headline, which is what let the margin grid be re-derived in #512
    *after* the headline was known. Selecting ``k`` on the split couples the
    two again.

    What replaces the guarantee is that the search space is **fixed in
    advance**. ``_SWEEP_TOP_K`` has been `(3, 5, 8, 15, 25)` since #511 and was
    not widened when it became a selection grid. Widening it after seeing a
    headline it now influences is precisely the fitting the split exists to
    prevent, so this asserts the sweep searches that grid and nothing else --
    if someone extends the constant, they must do it deliberately and
    re-register it, not discover it by watching a number improve.
    """
    report = _run(tmp_path, monkeypatch=monkeypatch, out_of_scope=True)

    rows = report["threshold_tuning"]["sweep"]
    assert {row["top_k"] for row in rows} == set(intent_index_eval._SWEEP_TOP_K)
    assert len(rows) == len(intent_index_eval._SWEEP_TOP_K) * len(
        intent_index_eval._SWEEP_MIN_MARGINS
    )
    selected = report["threshold_tuning"]["selected"]
    assert selected is None or selected["top_k"] in intent_index_eval._SWEEP_TOP_K


def test_the_sweep_breaks_accuracy_ties_toward_the_lower_top_k(tmp_path, monkeypatch):
    """The tie-break is what keeps a noise-sized gain from churning everything.

    Changing ``k`` re-measures every published number, so the pre-registered
    rule resolves ties toward lower ``k`` (and, before that, toward better
    out-of-scope deferral). Verified directly on the sweep rows rather than
    through a contrived fixture: among rows tied with the winner on served
    accuracy and deferral, none may carry a smaller ``k``.
    """
    report = _run(tmp_path, monkeypatch=monkeypatch, out_of_scope=True)
    selected = report["threshold_tuning"]["selected"]
    if selected is None:
        pytest.skip("no combination cleared the coverage floor on this fixture")

    tied = [
        row
        for row in report["threshold_tuning"]["sweep"]
        if row["coverage"] >= intent_index_eval._MIN_COVERAGE
        and row["served_accuracy"] == selected["served_accuracy"]
        and row["oos_deferral"] == selected["oos_deferral"]
    ]
    assert tied, "the selected row must appear among the eligible rows"
    assert selected["top_k"] == min(row["top_k"] for row in tied)


def test_the_module_sweep_never_changes_the_route(tmp_path, monkeypatch):
    """No ``min_module_score`` may move a single routing decision.

    ``_emit_modules`` runs *after* ``decide()`` has already taken its argmax
    over ``route_scores``, so the module threshold is structurally incapable of
    changing a route. That is the entire reason this threshold could be left
    dead at 0.45 for six PRs without any request being routed wrongly -- and
    the reason re-deriving it is safe to ship without re-litigating the
    headline. Asserting it beats assuming it: if module emission ever leaked
    into the routing path, every module number in the report would silently
    become a hyperparameter of the headline accuracy.
    """
    # Run the evaluation for its side effect of building the index on disk.
    _run(tmp_path, monkeypatch=monkeypatch, out_of_scope=True)
    index = IntentIndex.load(tmp_path / "index" / intent_index_eval.INDEX_FILENAME)

    def routes_at(min_module_score):
        return [
            index.decide(
                vector,
                min_margin=0.0,
                min_module_score=min_module_score,
                top_k=intent_index_eval.TOP_K,
            ).route
            for vector in index.vectors
        ]

    # 0.0 emits every module, 1.0 emits none and falls back to the single best
    # -- the two extremes the grid interpolates between.
    baseline = routes_at(intent_index_eval._DEFAULT_MIN_MODULE_SCORE)
    grid = intent_index_eval._module_score_grid(
        index, index.vectors, top_k=intent_index_eval.TOP_K
    )
    for min_module_score in grid + (0.0, 1.0):
        assert routes_at(min_module_score) == baseline, (
            f"min_module_score={min_module_score} moved a route"
        )

    # Non-vacuousness. Every route in the shared fixture carries exactly one
    # module, so emission there cannot vary with the threshold no matter what
    # the code does -- the loop above would pass even against a broken
    # implementation. A route with two modules is what makes the threshold
    # observable, and the route still must not move.
    # The two search modules must score *differently* or no threshold can
    # separate them: [0.8, 0.6, 0.0] is unit-norm and scores 0.8 against the
    # query, against lookup_fact's 1.0.
    off_axis = [0.8, 0.6, 0.0]
    multi = IntentIndex(
        examples=[
            CanonicalExample("m0", "m zero", "search", ("lookup_fact",)),
            CanonicalExample("m1", "m one", "search", ("lookup_document",)),
            CanonicalExample("m2", "m two", "chat", ("explain",)),
        ],
        vectors=np.array([_SEARCH, off_axis, _CHAT], dtype=np.float32),
        encoder="test",
        fingerprint="test",
    )
    query = np.array(_SEARCH, dtype=np.float32)
    wide = multi.decide(query, min_margin=0.0, min_module_score=0.0)
    narrow = multi.decide(query, min_margin=0.0, min_module_score=0.9)
    assert set(wide.modules) == {"lookup_fact", "lookup_document"}
    assert set(narrow.modules) == {"lookup_fact"}
    assert wide.route == narrow.route == "search"


def test_the_module_sweep_is_tuning_only_and_records_its_rule(tmp_path, monkeypatch):
    """The module threshold is chosen on the tuning slice, like the other two.

    Selection is highest macro-F1 with ties to the lower threshold, registered
    in advance. ``joint_accuracy`` is recorded on every row but must never be
    the selector: it is an exact-set match and peaks where emission collapses
    to the top-1 fallback, which macro-F1 correctly declines to choose.

    The grid is computed, not constant, so the assertion is on its *shape*: a
    legacy row plus ``_MODULE_GRID_STEPS`` derived ones. A hardcoded grid is
    what broke when ``top_k`` moved — at k=15 the k=3 constants excluded all
    but three candidate scores.
    """
    report = _run(tmp_path, monkeypatch=monkeypatch, out_of_scope=True)
    block = report["module_threshold_tuning"]

    assert block["tuned_on"] is True
    rows = block["sweep"]
    assert len(rows) == 1 + intent_index_eval._MODULE_GRID_STEPS
    assert rows[0]["min_module_score"] == intent_index_eval._LEGACY_MIN_MODULE_SCORE
    # Derived rows are ascending. Not asserted: that they exceed the legacy
    # row — true of the real e5 index, but this fixture's toy vectors score
    # arbitrarily low, and the grid must track whatever the data actually is.
    derived = [row["min_module_score"] for row in rows[1:]]
    assert derived == sorted(derived)
    selected = block["selected"]
    best = max(row["macro_f1"] for row in block["sweep"])
    assert selected["macro_f1"] == best
    # Ties resolve downward, so nothing below the winner may match its score.
    assert all(
        row["min_module_score"] >= selected["min_module_score"]
        for row in block["sweep"]
        if row["macro_f1"] == best
    )


# ---------------------------------------------------------------------------
# The pinned bars.
#
# Everything above runs on a synthetic index and no encoder. Everything below
# measures the *committed* canonical set with the real e5-small-v2 encoder, so
# it skips wherever sentence-transformers or the built index is absent -- which
# is every CI job, none of which installs sentence-transformers. These bars are
# a local guard, not a gate; nothing enforces them on a pull request.
#
# Raise a floor when a run beats it; never lower one without recording why in
# the commit message.
# ---------------------------------------------------------------------------

DATA = Path(__file__).resolve().parents[2] / "data"

# Re-measured 2026-08-14 on the intfloat/e5-small-v2 index over the 304-example
# canonical set, on the WIDER instrument (201-query test slice, 60 probes split
# into 29 tuning / 31 reporting), at the top_k that instrument selected:
#   test-slice route accuracy  0.8159 (201 queries, split seed 17) -> floor 0.79
#   out-of-scope AUC           0.8578 (31 held-out probes)         -> floor 0.83
#   p95 routing latency       12.20 ms                             -> ceiling 25.0 ms
#     (that bar now lives in tests/load/test_intent_routing_latency.py -- see #593)
#
# The accuracy floor is unchanged at 0.79 and the measurement rose slightly
# (0.8108 -> 0.8159) despite nearly doubling the slice.
#
# THE AUC FLOOR IS LOWERED, 0.85 -> 0.83, and that needs its reason recorded
# because the convention says never lower one silently. The number did not
# regress -- the measurement changed. AUC is now computed against the *reporting*
# half of the probes, which no sweep has ever seen. Previously the same 24
# probes both tie-broke threshold selection and denominated the reported AUC, so
# the old 0.8720 was measured partly on data that had selected the thresholds it
# was measured at. 0.8578 against held-out probes is the harder and more honest
# number, and 0.83 restores the ~0.02 headroom the convention asks for.
#
# Deliberately NOT pinned: hard_40 argmax. It is 40 queries with a 95% CI of
# roughly [0.55, 0.82] -- far too wide to floor without producing false alarms.
#
# The accuracy bar reads report["test_slice"], not report["bulk"]: `bulk` is
# the *mixed* tuning+test set, so a floor there would quietly re-admit the
# contamination the tuning/test split exists to remove.
#
# The out-of-scope bar is an AUC now, where it used to be the raw cosine
# margin (floor 0.10, from MiniLM's 0.1188). Raw margin is encoder-specific:
# e5 scores 0.0280 over the *same* probes while separating comparably, because
# it compresses cosines into a narrow high band. A raw-margin floor therefore
# rejects an encoder for its cosine range rather than for its separation. AUC
# and Cohen's d are scale-free; raw margin stays in the report as
# encoder-specific context only and must not be compared across encoders.
_TEST_SLICE_ACCURACY_FLOOR = 0.79
_OUT_OF_SCOPE_AUC_FLOOR = 0.83


@functools.lru_cache(maxsize=1)
def _report():
    pytest.importorskip("sentence_transformers")
    from src.model.pre_training.intents.model import DEFAULT_ENCODER
    from src.model.pre_training.intents.evaluation import run_index_evaluation
    from src.model.pre_training.intents.model import INDEX_FILENAME, IntentIndex

    index = DATA / "intent_index"
    if not (index / INDEX_FILENAME).exists():
        pytest.skip(
            "run `python -m src.model.pre_training.intents.cli build --canonical "
            f"data/intent_canonical.json --output {index}` to measure the bars"
        )
    on_disk_encoder = IntentIndex.load(index / INDEX_FILENAME).encoder
    if on_disk_encoder != DEFAULT_ENCODER:
        # run_index_evaluation now rejects this itself (an index whose
        # encoder differs from the one in use scores silent garbage
        # otherwise), so a stale on-disk index is a skip here rather than an
        # error: these bars are unmeasurable until the index is rebuilt.
        pytest.skip(
            f"data/intent_index was built with encoder {on_disk_encoder!r}, "
            f"but the current encoder is {DEFAULT_ENCODER!r}; run "
            "`python -m src.model.pre_training.intents.cli build --canonical "
            f"data/intent_canonical.json --output {index}` to re-measure "
            "the bars"
        )
    return run_index_evaluation(
        index_path=index,
        eval_queries_path=DATA / "intent_eval_queries.json",
        hard_queries_path=DATA / "intent_eval_hard.json",
        out_of_scope_path=DATA / "intent_out_of_scope.json",
        canonical_path=DATA / "intent_canonical.json",
        output_path=index / "evaluation_report.json",
    )


def test_index_holds_the_test_slice_accuracy_bar():
    """The untouched slice, never `bulk` -- see the constant's comment."""
    report = _report()
    assert report["test_slice"]["tuned_on"] is False
    assert report["test_slice"]["accuracy"] >= _TEST_SLICE_ACCURACY_FLOOR


def test_out_of_scope_requests_score_below_in_scope_requests():
    """Scale-free separability, so the bar survives the next encoder swap."""
    assert _report()["out_of_scope"]["auc"] >= _OUT_OF_SCOPE_AUC_FLOOR


def test_every_module_has_enough_canonical_support_to_be_emitted():
    """Canonical module support is a fact about the canonical set, not the
    encoder -- read it directly rather than through ``_report()``, so this
    stays measurable even while the on-disk index is stale and its encoder
    guard makes ``_report()`` skip.

    Mirrors ``IntentIndex.low_support_modules()`` exactly: every module
    across every route, counted from the canonical examples alone, compared
    against ``MIN_MODULE_SUPPORT``. No vectors, no ``IntentIndex`` instance,
    no encoder involved on either side of that computation.
    """
    from collections import Counter

    from src.model.pre_training.intents.data import load_canonical_examples
    from src.model.pre_training.intents.model import MIN_MODULE_SUPPORT
    from src.model.pre_training.intents.model import INTENT_LABELS, modules_for_route

    canonical_path = DATA / "intent_canonical.json"
    if not canonical_path.exists():
        pytest.skip(f"canonical example file is missing: {canonical_path}")
    examples = load_canonical_examples(canonical_path)
    counts = Counter(module for example in examples for module in example.modules)
    all_modules = [m for route in INTENT_LABELS for m in modules_for_route(route)]
    low_support = [m for m in all_modules if counts[m] < MIN_MODULE_SUPPORT]

    assert low_support == []


def test_the_report_covers_the_whole_bulk_set():
    """A silently shrunken eval set would inflate every later number.

    Eval-set size is a fact about ``data/intent_eval_queries.json``, not the
    encoder -- read it directly rather than through ``_report()``, for the
    same reason as the module-support test above.
    """
    from src.model.pre_training.intents.data import load_intent_eval_queries

    queries_path = DATA / "intent_eval_queries.json"
    if not queries_path.exists():
        pytest.skip(f"eval queries file is missing: {queries_path}")
    queries = load_intent_eval_queries(queries_path)
    legacy = [q for q in queries if q.id.startswith(intent_index_eval.LEGACY_PREFIX)]

    assert len(queries) >= 170
    assert len(legacy) == 30
