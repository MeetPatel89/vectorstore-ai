"""Embedding-provider factory shared by the retrieval demos."""

from __future__ import annotations

import hashlib
import re
from typing import override

from vectorstore import (
    EmbeddingProvider,
    EmbeddingSpec,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
)

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedding(EmbeddingProvider):
    """Small deterministic bag-of-words embedder for offline demonstrations.

    This provider is intentionally simple: it makes dense-search mechanics
    visible without an API key, network access, or a model download. It is not
    intended for production retrieval quality.
    """

    def __init__(
        self,
        dimension: int = 256,
        *,
        model: str = "blake2b-bow",
        version: str = "v1",
    ) -> None:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise ValueError("dimension must be a positive integer")
        self._spec = EmbeddingSpec(
            provider="demo-hash",
            model=model,
            dimension=dimension,
            version=version,
        )

    @property
    @override
    def spec(self) -> EmbeddingSpec:
        return self._spec

    @override
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimension
            vector[bucket] += 1.0
        return vector


def make_embedder(name: str = "hash") -> EmbeddingProvider:
    """Create an offline hash, OpenAI, or local sentence-transformer provider."""
    normalized = name.strip().lower()
    if normalized == "hash":
        return HashEmbedding()
    if normalized == "openai":
        return OpenAIEmbedding()
    if normalized == "local":
        return SentenceTransformerEmbedding()
    raise ValueError("provider must be one of: hash, openai, local")
