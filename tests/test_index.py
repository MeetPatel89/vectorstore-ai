from __future__ import annotations

from importlib.util import find_spec

import pytest
from conftest import FakeEmbedding

from vectorstore import (
    AzureSqlVectorStore,
    Chunk,
    FaissVectorStore,
    NumpyVectorStore,
    VectorIndex,
    create_store,
)


def test_index_end_to_end() -> None:
    embedder = FakeEmbedding(dimension=256)
    index = VectorIndex(embedder, NumpyVectorStore())
    chunks = [
        Chunk(
            "sso",
            "Users cannot login after certificate rotation",
            {"doc_type": "runbook"},
        ),
        Chunk(
            "payments",
            "Payment export totals differ from dashboard totals",
            {"doc_type": "known_issue"},
        ),
        Chunk(
            "limits",
            "API clients receive too many requests errors",
            {"doc_type": "runbook"},
        ),
    ]

    index.index(chunks)

    assert index.count() == 3
    assert embedder.document_calls == [[chunk.text for chunk in chunks]]
    assert index.search("certificate rotation login", k=1)[0].chunk.id == "sso"
    assert [
        result.chunk.id
        for result in index.search(
            "payment dashboard totals",
            k=3,
            filter={"doc_type": "known_issue"},
        )
    ] == ["payments"]
    assert embedder.query_calls == [
        "certificate rotation login",
        "payment dashboard totals",
    ]

    index.delete(["sso"])
    assert index.count() == 2
    assert "sso" not in {result.chunk.id for result in index.search("login", k=5)}


def test_empty_index_does_not_call_embedder() -> None:
    embedder = FakeEmbedding()
    index = VectorIndex(embedder, NumpyVectorStore())

    index.index([])
    assert index.search("unused", k=0) == []

    assert embedder.document_calls == []
    assert embedder.query_calls == []


def test_builtin_store_factory() -> None:
    assert isinstance(create_store("numpy", dimension=4), NumpyVectorStore)
    if find_spec("faiss") is None:
        with pytest.raises(ImportError, match="'faiss' extra"):
            create_store("faiss", dimension=4)
    else:
        assert isinstance(create_store("faiss", dimension=4), FaissVectorStore)
    assert isinstance(
        create_store("azure-sql", dimension=4, connection_factory=lambda: None),
        AzureSqlVectorStore,
    )


def test_index_dependencies_cannot_be_rebound_after_space_validation() -> None:
    index = VectorIndex(FakeEmbedding(dimension=2), NumpyVectorStore())

    with pytest.raises(AttributeError):
        setattr(index, "embedder", FakeEmbedding(dimension=3))
    with pytest.raises(AttributeError):
        setattr(index, "store", NumpyVectorStore(dimension=3))
