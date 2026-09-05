from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import override

import pytest

from vectorstore import (
    ChromaVectorStore,
    EmbeddingProvider,
    EmbeddingSpec,
    FaissVectorStore,
    NumpyVectorStore,
    VectorStore,
)


class FakeEmbedding(EmbeddingProvider):
    """A deterministic hashed bag-of-words embedder for offline tests."""

    def __init__(
        self,
        dimension: int = 64,
        provider: str = "fake",
        model: str = "hashed-bow",
    ) -> None:
        self._dimension = dimension
        self._provider = provider
        self._model = model
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    @override
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(
            provider=self._provider,
            model=self._model,
            dimension=self._dimension,
        )

    @override
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [self._embed(text) for text in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimension
            vector[bucket] += 1.0
        return vector


@pytest.fixture(params=["numpy", "chroma", "faiss"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> VectorStore:
    if request.param == "numpy":
        return NumpyVectorStore()
    if request.param == "chroma":
        pytest.importorskip("chromadb")
        return ChromaVectorStore(
            path=tmp_path / "chroma",
            collection_name="contract-tests",
        )
    pytest.importorskip("faiss")
    return FaissVectorStore()
