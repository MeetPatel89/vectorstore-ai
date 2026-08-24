"""Tests for SentenceTransformerEmbedding.

The real-model tests require the ``local`` extra (sentence-transformers +
torch) and download a small model on first run; they skip automatically
when the extra is not installed.
"""

import math

import pytest

st = pytest.importorskip("sentence_transformers")

from vectorstore import SentenceTransformerEmbedding  # noqa: E402


class TestSpec:
    def test_default_model_spec(self):
        provider = SentenceTransformerEmbedding()
        assert provider.spec.provider == "st"
        assert provider.spec.model == "all-MiniLM-L6-v2"
        assert provider.spec.dimension == 384
        assert provider.spec.space_id == "st__all_minilm_l6_v2__384__v1"

    def test_unknown_model_requires_explicit_dimension(self):
        with pytest.raises(ValueError, match="pass dimension explicitly"):
            SentenceTransformerEmbedding(model="some-unknown-model")
        provider = SentenceTransformerEmbedding(
            model="some-unknown-model", dimension=512
        )
        assert provider.spec.dimension == 512

    def test_constructing_does_not_load_the_model(self):
        provider = SentenceTransformerEmbedding(model="some-unknown-model-xyz", dimension=4)
        assert provider._loaded_model is None

    def test_rejects_invalid_arguments(self):
        with pytest.raises(ValueError):
            SentenceTransformerEmbedding(batch_size=0)
        with pytest.raises(ValueError):
            SentenceTransformerEmbedding(dimension=0)

    def test_distinct_space_from_openai(self):
        provider = SentenceTransformerEmbedding()
        assert not provider.spec.space_id.startswith("openai")


@pytest.fixture(scope="module")
def provider() -> SentenceTransformerEmbedding:
    return SentenceTransformerEmbedding()


@pytest.mark.local_model
class TestRealModel:

    def test_embed_texts_shape_and_order(self, provider):
        vectors = provider.embed_texts(["first text", "second text"])
        assert len(vectors) == 2
        assert all(len(vector) == 384 for vector in vectors)
        assert vectors[0] != vectors[1]

    def test_vectors_are_l2_normalized(self, provider):
        [vector] = provider.embed_texts(["normalize me"])
        norm = math.sqrt(sum(value * value for value in vector))
        assert norm == pytest.approx(1.0, abs=1e-3)

    def test_empty_input(self, provider):
        assert provider.embed_texts([]) == []

    def test_embed_query_matches_document_embedding(self, provider):
        query = provider.embed_query("payment reconciliation")
        [doc] = provider.embed_texts(["payment reconciliation"])
        assert query == pytest.approx(doc)
