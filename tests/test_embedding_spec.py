from __future__ import annotations

import openai
import pytest
from conftest import FakeEmbedding

from vectorstore import EmbeddingSpec, NumpyVectorStore, OpenAIEmbedding, VectorIndex


def test_space_id_is_sanitized_and_stable() -> None:
    spec = EmbeddingSpec(
        provider="openai",
        model="text-embedding-3-small",
        dimension=1536,
    )

    assert spec.space_id == "openai__text_embedding_3_small__1536__v1"
    assert spec.space_id == spec.space_id


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="provider"):
        EmbeddingSpec(provider="", model="m", dimension=8)
    with pytest.raises(ValueError, match="dimension"):
        EmbeddingSpec(provider="p", model="m", dimension=0)
    with pytest.raises(ValueError, match="version"):
        EmbeddingSpec(provider="p", model="m", dimension=8, version="")


def test_different_versions_produce_different_spaces() -> None:
    v1 = EmbeddingSpec(provider="p", model="m", dimension=8, version="v1")
    v2 = EmbeddingSpec(provider="p", model="m", dimension=8, version="v2")

    assert v1 != v2
    assert v1.space_id != v2.space_id


def test_openai_embedding_exposes_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: object())
    embedder = OpenAIEmbedding(api_key="test-key")

    assert embedder.spec == EmbeddingSpec(
        provider="openai",
        model="text-embedding-3-small",
        dimension=1536,
        version="v1",
    )
    assert embedder.dimension == 1536


def test_provider_dimension_defaults_to_spec_dimension() -> None:
    assert FakeEmbedding(dimension=32).dimension == 32


def test_vector_index_rejects_mismatched_store_dimension() -> None:
    embedder = FakeEmbedding(dimension=8)

    with pytest.raises(ValueError, match="one store per embedding space"):
        VectorIndex(embedder, NumpyVectorStore(dimension=16))


def test_vector_index_accepts_matching_store_and_exposes_spec() -> None:
    embedder = FakeEmbedding(dimension=8)

    index = VectorIndex(embedder, NumpyVectorStore(dimension=8))

    assert index.spec == embedder.spec
    assert index.spec.space_id == "fake__hashed_bow__8__v1"
