import subprocess
import sys

from src.internal.document_index.description_eval_cli import main
import pytest


def test_help_is_lazy():
    code = """
import sys
from src.internal.document_index.description_eval_cli import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "--preview-max-nodes" in run.stdout


def test_requires_existing_local_model(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--hierarchy",
                "unused",
                "--queries",
                "unused",
                "--output",
                str(tmp_path / "out.json"),
                "--model",
                str(tmp_path / "missing"),
            ]
        )
    assert error.value.code == 2
    assert "existing local model directory" in capsys.readouterr().err


def test_cli_atomic_report_with_injected_judge(tmp_path, monkeypatch):
    import json
    from src.internal.document_index.decision_model import LocalDecisionModel
    from src.internal.document_index.hierarchy import HierarchyConfig, build_hierarchy

    class Judge:
        def binary(self, query, candidates):
            return [1.0] * len(candidates)

        def choice(self, query, candidates):
            return [1 / len(candidates)] * len(candidates)

    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    loaded = []

    def load(path, **kwargs):
        loaded.append((path, kwargs))
        return Judge()

    monkeypatch.setattr(LocalDecisionModel, "from_pretrained", load)
    tree = build_hierarchy(
        [
            {
                "id": "a",
                "title": "Alpha",
                "contents": "Evidence",
                "metadata": {"path": ["Root"]},
            }
        ],
        HierarchyConfig(mode="structure"),
    )
    hierarchy = tmp_path / "tree.json"
    tree.save(hierarchy)
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {"id": "q", "query": "Find evidence", "gold_passage_ids": ["a::passage-0"]}
        )
        + "\n"
    )
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--hierarchy",
                str(hierarchy),
                "--queries",
                str(queries),
                "--model",
                str(model_dir),
                "--output",
                str(output),
                "--topk",
                "1",
                "--bootstrap-samples",
                "10",
                "--passage-words",
                "17",
                "--max-passage-chars",
                "240",
                "--batch-size",
                "3",
                "--min-relevance",
                "0.25",
                "--inference-batch-size",
                "2",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert set(report["variants"]) == {"stored", "label_only", "extractive"}
    assert (
        report["variants"]["stored"]["macro_metrics"]["final_recall_at_topk"]["mean"]
        == 1
    )
    assert len(loaded) == 1
    assert loaded[0][1]["inference_batch_size"] == 2
    assert report["config"]["passage_words"] == 17
    assert report["config"]["max_passage_chars"] == 240
    assert report["config"]["batch_size"] == 3
    assert report["config"]["min_relevance"] == 0.25
    assert not list(tmp_path.glob("*.tmp"))
