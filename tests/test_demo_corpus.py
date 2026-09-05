from __future__ import annotations

from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

import pytest

# Demos are scripts whose siblings are top-level modules, not a Python package.
corpus = import_module("examples._corpus")


def test_demo_default_is_a_bundled_synthetic_corpus() -> None:
    parser = ArgumentParser()
    corpus.add_corpus_argument(parser)
    root = parser.parse_args([]).corpus
    assert root == corpus.CORPUS_ROOT
    documents = corpus.load_documents(root)
    assert len(documents) == 6
    assert {doc.visibility for doc, _, _ in documents} == {
        "internal",
        "customer_safe",
    }
    assert {"INC-1104", "CHG-2407"} <= {doc.doc_id for doc, _, _ in documents}


def test_custom_corpus_and_invalid_paths(tmp_path: Path) -> None:
    parser = ArgumentParser()
    corpus.add_corpus_argument(parser)
    for root in (tmp_path, tmp_path / "missing"):
        with pytest.raises(SystemExit) as error:
            parser.parse_args(["--corpus", str(root)])
        assert error.value.code == 2
    (tmp_path / "custom.md").write_text(
        "---\ndoc_id: CUSTOM-1\ntitle: Custom example\n---\n\nA synthetic body.\n"
    )
    root = parser.parse_args(["--corpus", str(tmp_path)]).corpus
    assert root == tmp_path.resolve()
    assert corpus.load_documents(root)[0][0].doc_id == "CUSTOM-1"
