"""Multi-day churn: incremental updates stay proportional to change."""

import random
import subprocess
import sys

from examples.simulate_hierarchy_churn import simulate
from src.internal.document_index.decision_retriever import DecisionRetrieverConfig
from src.internal.document_index.description_eval import evaluate_descriptions
from src.internal.document_index.hierarchy import HierarchyConfig
from tests.unit.hierarchy_fakes import CountingDescriber, TopicEmbedder, docs

EXISTING = ("alpha", "beta", "delta")


class TopicDescriber(CountingDescriber):
    """Counting describer whose text names the topics of its documents."""

    def describe_with_provenance(self, label, documents, *, vectors=None, sample_size):
        result = super().describe_with_provenance(
            label, documents, vectors=vectors, sample_size=sample_size
        )
        topics = sorted({d["title"].split()[0] for d in documents})
        result["text"] = "Notes about " + " ".join(topics)
        return result


class KeywordJudge:
    """Scores a candidate by the fraction of query words it contains."""

    def binary(self, query, candidates):
        words = set(query.lower().split())
        return [
            len(words & set(c.text.lower().replace(".", " ").split())) / len(words)
            for c in candidates
        ]

    def choice(self, query, candidates):
        scores = self.binary(query, candidates)
        total = sum(scores) or 1.0
        return [s / total for s in scores]


def schedule():
    rng = random.Random(7)
    initial = sum((docs(t, 30) for t in EXISTING), [])
    alive = [d["id"] for d in initial]
    next_index = dict.fromkeys(EXISTING, 30)
    days = []
    for day in range(1, 6):
        added = []
        for k in range(5):
            topic = EXISTING[(day + k) % 3]
            added += docs(topic, 1, start=next_index[topic])
            next_index[topic] += 1
        deleted = rng.sample(alive, 3)
        if day == 3:
            added += docs("gamma", 10)
        alive = [i for i in alive if i not in deleted] + [d["id"] for d in added]
        days.append((added, deleted))
    return initial, days


def test_five_day_churn_costs_less_than_rebuilds_and_routes_new_topic():
    initial, days = schedule()
    config = HierarchyConfig(
        mode="cluster", n_clusters=3, description_model="counting-describer"
    )
    incremental = (TopicEmbedder(), TopicDescriber())
    rebuild = (TopicEmbedder(), TopicDescriber())
    result = simulate(initial, days, config, incremental=incremental, rebuild=rebuild)

    sizes = [len(initial)]
    for added, deleted in days:
        sizes.append(sizes[-1] + len(added) - len(deleted))
    total_added = sum(len(added) for added, _ in days)
    cumulative = result["cumulative"]
    assert cumulative["incremental"]["embedded_documents"] == 90 + total_added
    assert sum(incremental[0].calls) == 90 + total_added
    assert cumulative["rebuild"]["embedded_documents"] == sum(sizes)
    assert sum(rebuild[0].calls) == sum(sizes)
    assert cumulative["incremental"]["description_model_calls"] == len(
        incremental[1].calls
    )
    assert cumulative["rebuild"]["description_model_calls"] == len(rebuild[1].calls)
    assert (
        cumulative["incremental"]["description_model_calls"]
        < cumulative["rebuild"]["description_model_calls"]
    )
    assert len(result["days"]) == 6

    spawns = [
        (record["day"], op)
        for record in result["days"]
        for op in record["incremental"]["structure_operations"]
        if op["op"] == "spawn"
    ]
    assert len(spawns) == 1 and spawns[0][0] == 3
    final = result["incremental_artifact"]
    gamma = {d["id"] for d in docs("gamma", 10)}
    gamma_groups = {final.assignments[i] for i in gamma}
    assert gamma_groups == {spawns[0][1]["group_ids"][0]}
    spawned = next(g for g in final.groups if g["id"] in gamma_groups)
    assert set(spawned["document_ids"]) == gamma

    day0 = {g["id"] for g in result["day0_groups"]}
    touched = {
        gid
        for record in result["days"]
        for op in record["incremental"]["structure_operations"]
        if op["op"] in ("remove", "split", "merge")
        for gid in op["source_ids"]
    }
    untouched = day0 - touched
    assert untouched
    assert untouched <= {g["id"] for g in final.groups}

    report = evaluate_descriptions(
        final,
        KeywordJudge(),
        [
            {
                "id": "gamma",
                "query": "gamma document 3",
                "gold_passage_ids": ["gamma-003::passage-0"],
            }
        ],
        {},
        DecisionRetrieverConfig(beam_width=1),
        topk=1,
        bootstrap_samples=5,
    )
    record = report["variants"]["stored"]["queries"][0]
    assert record["metrics"]["complete_path_survival"] == 1.0


def test_simulation_script_help_loads_no_models():
    code = """
import sys
from examples.simulate_hierarchy_churn import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'sentence_transformers' not in sys.modules
"""
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "--daily-delete" in run.stdout
    assert "no pass threshold" in run.stdout


def test_simulation_cli_writes_counts_and_skips_absent_gold(tmp_path, monkeypatch):
    import json

    from examples.simulate_hierarchy_churn import main
    from src.internal.document_index import decision_model, hierarchy_models

    class Embedder(TopicEmbedder):
        def __init__(self, model_name, device="cpu"):
            super().__init__()
            self.model_name = model_name

    class Describer(TopicDescriber):
        def __init__(self, model_name, device="cpu"):
            super().__init__()
            self.model_name = model_name

    monkeypatch.setattr(hierarchy_models, "SentenceEmbedding", Embedder)
    monkeypatch.setattr(hierarchy_models, "LocalHierarchyDescriber", Describer)
    monkeypatch.setattr(
        decision_model.LocalDecisionModel,
        "from_pretrained",
        staticmethod(lambda path, **kwargs: KeywordJudge()),
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps(d) + "\n" for t in EXISTING for d in docs(t, 20))
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {"id": "gone", "query": "x", "gold_passage_ids": ["missing::passage-0"]}
        )
        + "\n"
    )
    output = tmp_path / "out" / "churn.json"
    main(
        [
            "--corpus-path", str(corpus), "--days", "2", "--daily-add", "4",
            "--daily-delete", "2", "--embedding-model", "embed",
            "--description-model", "describe", "--decision-model", "judge",
            "--queries", str(queries), "--output", str(output), "--seed", "1",
        ]
    )  # fmt: skip
    report = json.loads(output.read_text())
    assert "no pass threshold" in report["note"]
    assert [d["corpus_size"] for d in report["days"]] == [52, 54, 56]
    cumulative = report["cumulative"]
    assert cumulative["incremental"]["embedded_documents"] == 52 + 8
    assert cumulative["rebuild"]["embedded_documents"] == 52 + 54 + 56
    for track in ("incremental", "rebuild"):
        assert report["evaluation"][track] == {"skipped": ["gone"], "report": None}
