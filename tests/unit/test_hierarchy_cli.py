import json
import subprocess
import sys

import pytest

from src.internal.document_index.hierarchy import HierarchyArtifact
from src.internal.document_index.hierarchy_cli import main


def corpus(path, count=2):
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": str(i),
                    "title": f"Title {i}",
                    "text": "Body text.",
                    "metadata": {"hierarchy_path": ["Guide"]},
                }
            )
            for i in range(count)
        )
    )


def test_help_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "src.internal.document_index.hierarchy_cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "build" in result.stdout
    assert "inspect" in result.stdout


def test_offline_build_update_inspect(tmp_path, capsys, monkeypatch):
    source = tmp_path / "docs.jsonl"
    artifact = tmp_path / "tree.json"
    corpus(source)
    import src.internal.document_index.hierarchy_models as models

    monkeypatch.setattr(
        models,
        "SentenceEmbedding",
        lambda *a, **kw: pytest.fail("structure corpus initialized encoder"),
    )
    assert (
        main(
            [
                "build",
                "--corpus-path",
                str(source),
                "--output",
                str(artifact),
                "--clusters",
                "1000",
            ]
        )
        == 0
    )
    assert HierarchyArtifact.load(artifact).config["n_clusters"] == 1000
    corpus(source, 1)
    assert (
        main(
            [
                "update",
                "--corpus",
                str(source),
                "--previous",
                str(artifact),
                "--output",
                str(artifact),
            ]
        )
        == 0
    )
    assert main(["inspect", str(artifact)]) == 0
    output = capsys.readouterr().out.strip().splitlines()
    assert json.loads(output[-1])["deleted_documents"] == 1
    assert len(HierarchyArtifact.load(artifact).documents) == 1


def test_bad_corpus_error_is_clean(tmp_path, capsys):
    source = tmp_path / "bad.jsonl"
    source.write_text("{bad json")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "build",
                "--corpus-path",
                str(source),
                "--output",
                str(tmp_path / "tree.json"),
            ]
        )
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_update_rejects_encoder_override(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "update",
                "--corpus",
                "x",
                "--previous",
                "x",
                "--output",
                "x",
                "--embedding-model",
                "different",
            ]
        )
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_build_exposes_description_sample_budget(tmp_path):
    import json
    from src.internal.document_index.hierarchy_cli import main
    from src.internal.document_index.hierarchy import HierarchyArtifact

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": str(i),
                    "title": f"Topic {i}",
                    "contents": "body",
                    "metadata": {"hierarchy_path": ["Root"]},
                }
            )
            for i in range(3)
        )
    )
    output = tmp_path / "tree.json"
    assert (
        main(
            [
                "build",
                "--mode",
                "structure",
                "--corpus-path",
                str(corpus),
                "--output",
                str(output),
                "--description-sample-size",
                "2",
            ]
        )
        == 0
    )
    artifact = HierarchyArtifact.load(output)
    assert artifact.config["description_sample_size"] == 2
    assert len(artifact.groups[0]["description_provenance"]["document_ids"]) == 2


def test_build_stores_maintenance_flags(tmp_path):
    source = tmp_path / "docs.jsonl"
    output = tmp_path / "tree.json"
    corpus(source)
    assert (
        main(
            [
                "build",
                "--mode",
                "structure",
                "--corpus-path",
                str(source),
                "--output",
                str(output),
                "--min-cluster-size",
                "3",
                "--max-fanout",
                "4",
            ]
        )
        == 0
    )
    config = HierarchyArtifact.load(output).config
    assert config["min_cluster_size"] == 3
    assert config["max_fanout"] == 4


def test_update_dry_run_prints_diagnostics_and_writes_nothing(tmp_path, capsys):
    source = tmp_path / "docs.jsonl"
    artifact = tmp_path / "tree.json"
    output = tmp_path / "dry.json"
    corpus(source)
    assert main(["build", "--corpus-path", str(source), "--output", str(artifact)]) == 0
    capsys.readouterr()
    corpus(source, 3)
    assert (
        main(
            [
                "update",
                "--corpus-path",
                str(source),
                "--previous",
                str(artifact),
                "--output",
                str(output),
                "--dry-run",
                "--max-description-calls",
                "0",
            ]
        )
        == 0
    )
    assert not output.exists()
    diagnostics = json.loads(capsys.readouterr().out)
    assert diagnostics["dry_run"] is True
    assert "structure_operations" in diagnostics
    assert diagnostics["update_cost"]["description_model_calls"] == 0


def test_update_passes_description_call_cap(tmp_path, monkeypatch):
    import src.internal.document_index.hierarchy_cli as cli

    source = tmp_path / "docs.jsonl"
    artifact = tmp_path / "tree.json"
    corpus(source)
    assert main(["build", "--corpus-path", str(source), "--output", str(artifact)]) == 0
    seen = {}
    real = cli.build_hierarchy

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "build_hierarchy", spy)
    args = ["--corpus-path", str(source), "--previous", str(artifact)]
    assert (
        main(
            ["update", *args, "--output", str(artifact), "--max-description-calls", "3"]
        )
        == 0
    )
    assert seen["max_description_calls"] == 3
    assert seen["dry_run"] is False


def test_update_help_documents_dry_run_cost():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.internal.document_index.hierarchy_cli",
            "update",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    text = " ".join(result.stdout.split())
    assert "--max-description-calls" in text
    assert "new documents are still embedded" in text
    assert "no description model calls are made" in text


def test_build_has_no_update_only_flags(capsys):
    with pytest.raises(SystemExit):
        main(["build", "--corpus", "x", "--output", "x", "--dry-run"])
    assert "unrecognized arguments" in capsys.readouterr().err
