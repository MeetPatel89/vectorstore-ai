from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vectorstore import Chunk, NumpyVectorStore


def test_save_load_round_trip(tmp_path: Path) -> None:
    store = NumpyVectorStore()
    chunks = [
        Chunk("one", "First", {"kind": "note"}),
        Chunk("two", "Second", {"rank": 2}),
    ]
    store.upsert(chunks, [[3.0, 0.0], [0.0, 4.0]])
    store.save(tmp_path / "saved")

    loaded = NumpyVectorStore.load(tmp_path / "saved")

    assert loaded.dimension == 2
    assert loaded.count() == 2
    assert loaded.get(["one", "two"]) == chunks
    assert loaded.search([1.0, 0.0], k=2)[0].chunk.id == "one"


def test_empty_store_with_known_dimension_round_trips(tmp_path: Path) -> None:
    store = NumpyVectorStore(dimension=7)
    store.save(tmp_path)

    loaded = NumpyVectorStore.load(tmp_path)

    assert loaded.dimension == 7
    assert loaded.count() == 0
    assert loaded.search([0.0] * 7) == []


def test_dimension_validation_on_upsert_and_search() -> None:
    store = NumpyVectorStore(dimension=3)

    with pytest.raises(ValueError, match="store dimension"):
        store.upsert([Chunk("bad", "Bad")], [[1.0, 2.0]])

    store.upsert([Chunk("good", "Good")], [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="store dimension"):
        store.search([1.0, 2.0])


def test_load_rejects_matrix_that_disagrees_with_saved_dimension(
    tmp_path: Path,
) -> None:
    np.savez_compressed(
        tmp_path / "vectors.npz",
        vectors=np.zeros((1, 2), dtype=np.float32),
        ids=np.asarray(["one"], dtype=np.str_),
        dimension=np.asarray(3, dtype=np.int64),
    )
    (tmp_path / "chunks.json").write_text(
        json.dumps([{"id": "one", "text": "One", "metadata": {}}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="saved vector dimension"):
        NumpyVectorStore.load(tmp_path)


def test_zero_vectors_are_safe_and_have_zero_similarity() -> None:
    store = NumpyVectorStore()
    store.upsert([Chunk("zero", "Zero")], [[0.0, 0.0]])

    assert store.search([1.0, 0.0])[0].score == 0.0
    assert store.search([0.0, 0.0])[0].score == 0.0
