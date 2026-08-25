"""OpenAI embedding provider."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, override

from .base import EmbeddingProvider, EmbeddingSpec

_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbedding(EmbeddingProvider):
    """Generate embeddings with OpenAI's embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 128,
        dimensions: int | None = None,
        version: str = "v1",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if dimensions is not None and dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is required; pass api_key or set OPENAI_API_KEY"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError("OpenAIEmbedding requires the 'openai' package") from exc

        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.version = version
        # The SDK retries rate limits, connection failures, and transient server
        # errors with exponential backoff while keeping this provider thin.
        self._client = OpenAI(api_key=resolved_key, max_retries=2)

    @property
    @override
    def spec(self) -> EmbeddingSpec:
        """Describe the provider's embedding space."""
        return EmbeddingSpec(
            provider="openai",
            model=self.model,
            dimension=self.dimension,
            version=self.version,
        )

    @property
    @override
    def dimension(self) -> int:
        """The number of elements in each produced embedding."""
        if self.dimensions is not None:
            return self.dimensions
        try:
            return _MODEL_DIMENSIONS[self.model]
        except KeyError as exc:
            raise ValueError(
                f"unknown embedding dimension for model {self.model!r}; "
                "pass dimensions explicitly"
            ) from exc

    @override
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts through the OpenAI embeddings API."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            request: dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions

            response = self._client.embeddings.create(**request)
            for i in response.data:
                print("--------------------------------")
                print(i)
                print("--------------------------------")
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise RuntimeError(
                    "OpenAI returned a different number of embeddings than requested"
                )
            embeddings.extend(_as_float_list(item.embedding) for item in ordered)

        return embeddings


def _as_float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]
