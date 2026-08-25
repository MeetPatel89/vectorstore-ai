from __future__ import annotations

from pathlib import Path

from vectorstore import ChromaVectorStore, Chunk
from vectorstore.stores.chroma_store import _translate_filter


def test_persistence_across_client_reopen(tmp_path: Path) -> None:
    path = tmp_path / "persistent-chroma"
    first = ChromaVectorStore(path=path, collection_name="persistence-test")
    first.upsert([Chunk("one", "Persistent", {"kind": "note"})], [[1.0, 0.0]])

    reopened = ChromaVectorStore(path=path, collection_name="persistence-test")

    assert reopened.count() == 1
    assert reopened.get(["one"]) == [Chunk("one", "Persistent", {"kind": "note"})]


def test_empty_metadata_is_accepted_and_returned_as_empty(tmp_path: Path) -> None:
    store = ChromaVectorStore(path=tmp_path, collection_name="empty-metadata-test")

    store.upsert(
        [Chunk("empty", "Had metadata", {"old": "value"})],
        [[0.0, 1.0]],
    )
    store.upsert([Chunk("empty", "No metadata")], [[1.0, 0.0]])

    assert store.get(["empty"]) == [Chunk("empty", "No metadata")]
    assert store.search([1.0, 0.0], k=1)[0].chunk.metadata == {}


def test_multi_key_filter_uses_explicit_and(tmp_path: Path) -> None:
    assert _translate_filter({"kind": "runbook", "priority": {"$gte": 2}}) == {
        "$and": [
            {"kind": {"$eq": "runbook"}},
            {"priority": {"$gte": 2}},
        ]
    }

    store = ChromaVectorStore(path=tmp_path, collection_name="and-filter-test")
    store.upsert(
        [
            Chunk("match", "Match", {"kind": "runbook", "priority": 3}),
            Chunk("wrong-kind", "Other", {"kind": "policy", "priority": 3}),
            Chunk("wrong-priority", "Other", {"kind": "runbook", "priority": 1}),
        ],
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    )

    results = store.search(
        [1.0, 0.0],
        k=3,
        filter={"kind": "runbook", "priority": {"$gte": 2}},
    )
    assert [result.chunk.id for result in results] == ["match"]
