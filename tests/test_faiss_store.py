from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

import pytest

from vectorstore import Chunk, FaissVectorStore

pytestmark = pytest.mark.skipif(find_spec("faiss") is None, reason="faiss extra")


def test_save_load_round_trip(tmp_path: Path) -> None:
    store = FaissVectorStore()
    chunks = [
        Chunk("one", "First", {"kind": "note"}),
        Chunk("two", "Second", {"rank": 2}),
    ]
    store.upsert(chunks, [[3.0, 0.0], [0.0, 4.0]])
    store.delete(["two"])
    store.upsert([Chunk("two", "Updated", {"rank": 3})], [[0.0, 1.0]])
    store.save(tmp_path / "saved")

    loaded = FaissVectorStore.load(tmp_path / "saved")

    assert loaded.dimension == 2
    assert loaded.count() == 2
    assert loaded.get(["one", "two"]) == [
        chunks[0],
        Chunk("two", "Updated", {"rank": 3}),
    ]
    assert loaded.search([1.0, 0.0], k=2)[0].chunk.id == "one"


def test_empty_store_with_known_dimension_round_trips(tmp_path: Path) -> None:
    store = FaissVectorStore(dimension=7)
    store.save(tmp_path)

    loaded = FaissVectorStore.load(tmp_path)

    assert loaded.dimension == 7
    assert loaded.count() == 0
    assert loaded.search([0.0] * 7) == []


def test_empty_store_without_dimension_round_trips(tmp_path: Path) -> None:
    store = FaissVectorStore()
    store.save(tmp_path)

    loaded = FaissVectorStore.load(tmp_path)

    assert loaded.dimension is None
    assert loaded.count() == 0
    assert loaded.search([1.0, 0.0]) == []


def test_dimension_validation_on_upsert_and_search() -> None:
    store = FaissVectorStore(dimension=3)

    with pytest.raises(ValueError, match="store dimension"):
        store.upsert([Chunk("bad", "Bad")], [[1.0, 2.0]])

    store.upsert([Chunk("good", "Good")], [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="store dimension"):
        store.search([1.0, 2.0])


def test_zero_vectors_are_safe_and_have_zero_similarity() -> None:
    store = FaissVectorStore()
    store.upsert([Chunk("zero", "Zero")], [[0.0, 0.0]])

    assert store.search([1.0, 0.0])[0].score == 0.0
    assert store.search([0.0, 0.0])[0].score == 0.0


def test_load_rejects_index_metadata_dimension_mismatch(tmp_path: Path) -> None:
    store = FaissVectorStore(dimension=2)
    store.save(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dimension"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        FaissVectorStore.load(tmp_path)
