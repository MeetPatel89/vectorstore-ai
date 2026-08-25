from __future__ import annotations

import pytest

from vectorstore import Chunk, VectorStore


def _seed(store: VectorStore) -> None:
    store.upsert(
        [
            Chunk(
                "alpha",
                "Alpha runbook",
                {"doc_type": "runbook", "priority": 3, "active": True},
            ),
            Chunk(
                "beta",
                "Beta issue",
                {"doc_type": "known_issue", "priority": 2, "active": True},
            ),
            Chunk(
                "gamma",
                "Gamma policy",
                {"doc_type": "policy", "priority": 1, "active": False},
            ),
        ],
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )


def test_upsert_replaces_vector_text_and_metadata(store: VectorStore) -> None:
    _seed(store)
    store.upsert(
        [Chunk("alpha", "Replaced article", {"doc_type": "policy", "priority": 8})],
        [[0.0, 0.0, 1.0]],
    )

    assert store.count() == 3
    assert store.get(["alpha"]) == [
        Chunk("alpha", "Replaced article", {"doc_type": "policy", "priority": 8})
    ]
    assert store.search([0.0, 0.0, 1.0], k=1)[0].chunk.id == "alpha"
    assert store.search([1.0, 0.0, 0.0], filter={"priority": 8})[0].chunk.id == "alpha"


def test_upsert_duplicate_id_uses_last_value(store: VectorStore) -> None:
    store.upsert(
        [Chunk("same", "first"), Chunk("same", "last", {"version": 2})],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert store.count() == 1
    assert store.get(["same"]) == [Chunk("same", "last", {"version": 2})]
    assert store.search([0.0, 1.0], k=1)[0].chunk.id == "same"


def test_delete_existing_and_nonexistent_ids(store: VectorStore) -> None:
    _seed(store)
    store.delete(["beta", "does-not-exist"])
    store.delete(["does-not-exist"])

    assert store.count() == 2
    assert store.get(["beta"]) == []
    assert "beta" not in {
        result.chunk.id for result in store.search([0.8, 0.6, 0.0], k=10)
    }


def test_search_is_descending_and_honors_top_k(store: VectorStore) -> None:
    _seed(store)
    results = store.search([1.0, 0.0, 0.0], k=2)

    assert [result.chunk.id for result in results] == ["alpha", "beta"]
    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert all(-1.0 <= result.score <= 1.0 for result in results)


@pytest.mark.parametrize(
    ("filter", "expected_ids"),
    [
        ({"doc_type": "runbook"}, {"alpha"}),
        ({"doc_type": {"$in": ["runbook", "known_issue"]}}, {"alpha", "beta"}),
        ({"priority": {"$gt": 1}}, {"alpha", "beta"}),
        ({"priority": {"$gte": 2}}, {"alpha", "beta"}),
        ({"priority": {"$lt": 2}}, {"gamma"}),
        ({"priority": {"$lte": 2}}, {"beta", "gamma"}),
        ({"active": True, "priority": {"$gte": 3}}, {"alpha"}),
    ],
)
def test_metadata_filter_operators(
    store: VectorStore, filter: dict[str, object], expected_ids: set[str]
) -> None:
    _seed(store)
    results = store.search([1.0, 0.0, 0.0], k=10, filter=filter)

    assert {result.chunk.id for result in results} == expected_ids


def test_filter_is_applied_before_top_k(store: VectorStore) -> None:
    _seed(store)
    results = store.search(
        [1.0, 0.0, 0.0],
        k=2,
        filter={"doc_type": {"$in": ["known_issue", "policy"]}},
    )

    assert [result.chunk.id for result in results] == ["beta", "gamma"]


def test_empty_store_and_nonpositive_k_return_no_results(store: VectorStore) -> None:
    assert store.search([1.0, 0.0], k=5) == []
    store.upsert([Chunk("one", "One")], [[1.0, 0.0]])
    assert store.search([1.0, 0.0], k=0) == []


def test_get_preserves_requested_order_and_skips_unknown_ids(
    store: VectorStore,
) -> None:
    _seed(store)

    chunks = store.get(["gamma", "missing", "alpha", "gamma"])

    assert [chunk.id for chunk in chunks] == ["gamma", "alpha", "gamma"]


def test_empty_mutations_are_noops(store: VectorStore) -> None:
    store.upsert([], [])
    store.delete([])

    assert store.count() == 0


def test_upsert_rejects_mismatched_lengths(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="same length"):
        store.upsert([Chunk("one", "One")], [])
