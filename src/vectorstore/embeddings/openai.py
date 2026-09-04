"""OpenAI embedding provider."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol, cast, override

from .base import (
    EmbeddingProvider,
    EmbeddingResult,
    EmbeddingSpec,
    EmbeddingUsage,
)
from .tokenization import (
    TokenCountingUnavailableError,
    estimate_tokens,
    token_counts,
)

_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_MAX_INPUTS_PER_REQUEST = 2_048
_MAX_TOKENS_PER_INPUT = 8_192
_MAX_TOKENS_PER_REQUEST = 300_000


class OpenAIEmbeddingsResource(Protocol):
    """The narrow SDK resource used by :class:`OpenAIEmbedding`."""

    def create(self, **kwargs: object) -> Any:
        """Create one embeddings response."""


class OpenAIClient(Protocol):
    """The injectable portion of an OpenAI client used by this provider."""

    @property
    def embeddings(self) -> OpenAIEmbeddingsResource:
        """The SDK embeddings resource."""


class OpenAIEmbedding(EmbeddingProvider):
    """Generate embeddings with OpenAI's embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 128,
        dimensions: int | None = None,
        version: str = "v1",
        encoding_name: str | None = None,
        client: OpenAIClient | None = None,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be greater than zero")
        if batch_size > _MAX_INPUTS_PER_REQUEST:
            raise ValueError(f"batch_size must not exceed {_MAX_INPUTS_PER_REQUEST}")
        if dimensions is not None and (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or dimensions <= 0
        ):
            raise ValueError("dimensions must be greater than zero")
        if not isinstance(version, str) or not version:
            raise ValueError("version must be a non-empty string")
        if encoding_name is not None and (
            not isinstance(encoding_name, str) or not encoding_name
        ):
            raise ValueError("encoding_name must be a non-empty string or None")

        self._model = model
        self._batch_size = batch_size
        self._dimensions = dimensions
        self._version = version
        self._encoding_name = encoding_name
        self._client = client if client is not None else _default_client(api_key)

    @property
    def model(self) -> str:
        """The immutable model identifier for this embedding space."""
        return self._model

    @property
    def batch_size(self) -> int:
        """The immutable maximum number of inputs in one provider batch."""
        return self._batch_size

    @property
    def dimensions(self) -> int | None:
        """The configured output-width override, when present."""
        return self._dimensions

    @property
    def version(self) -> str:
        """The immutable embedding-configuration version."""
        return self._version

    @property
    def encoding_name(self) -> str | None:
        """The explicit tokenizer encoding override, when present."""
        return self._encoding_name

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
    def estimate_tokens(self, texts: list[str]) -> int:
        """Count inputs with this model's tiktoken encoding."""
        return estimate_tokens(
            texts,
            self.model,
            encoding_name=self.encoding_name,
        )

    @override
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts through the OpenAI embeddings API."""
        return self.embed_texts_with_usage(texts).vectors

    @override
    def embed_texts_with_usage(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts and aggregate authoritative usage across API batches."""
        if not texts:
            return EmbeddingResult(vectors=[], usage=EmbeddingUsage(total_tokens=0))

        embeddings: list[list[float]] = []
        total_tokens = 0
        for batch in self._batches(texts):
            request: dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions

            response = self._client.embeddings.create(**request)
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise RuntimeError(
                    "OpenAI returned a different number of embeddings than requested"
                )
            embeddings.extend(_as_float_list(item.embedding) for item in ordered)
            try:
                batch_tokens = response.usage.prompt_tokens
            except AttributeError as exc:
                raise RuntimeError(
                    "OpenAI embedding response did not include valid token usage"
                ) from exc
            if (
                not isinstance(batch_tokens, int)
                or isinstance(batch_tokens, bool)
                or batch_tokens < 0
            ):
                raise RuntimeError(
                    "OpenAI embedding response did not include valid token usage"
                )
            total_tokens += batch_tokens

        return EmbeddingResult(
            vectors=embeddings,
            usage=EmbeddingUsage(total_tokens=total_tokens),
        )

    @override
    def embed_query_with_usage(self, text: str) -> EmbeddingResult:
        """Embed one query and retain the API's authoritative token usage."""
        return self.embed_texts_with_usage([text])

    def _batches(self, texts: list[str]) -> list[list[str]]:
        for index, text in enumerate(texts):
            if text == "":
                raise ValueError(f"embedding input at index {index} must not be empty")

        try:
            counts = token_counts(
                texts,
                self.model,
                encoding_name=self.encoding_name,
            )
        except TokenCountingUnavailableError:
            if self.encoding_name is not None:
                raise
            return [
                texts[offset : offset + self.batch_size]
                for offset in range(0, len(texts), self.batch_size)
            ]

        batches: list[list[str]] = []
        batch: list[str] = []
        batch_tokens = 0
        for index, (text, count) in enumerate(zip(texts, counts, strict=True)):
            if count > _MAX_TOKENS_PER_INPUT:
                raise ValueError(
                    f"embedding input at index {index} has {count} tokens; "
                    f"maximum is {_MAX_TOKENS_PER_INPUT}"
                )
            if batch and (
                len(batch) >= self.batch_size
                or batch_tokens + count > _MAX_TOKENS_PER_REQUEST
            ):
                batches.append(batch)
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += count

        if batch:
            batches.append(batch)
        return batches


def _as_float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _default_client(api_key: str | None) -> OpenAIClient:
    """Create the default SDK client when an application did not inject one."""
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "OpenAI API key is required; pass api_key or set OPENAI_API_KEY"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError("OpenAIEmbedding requires the 'openai' package") from exc
    # The SDK retries rate limits, connection failures, and transient server
    # errors with exponential backoff while keeping this provider thin.
    return cast(OpenAIClient, OpenAI(api_key=resolved_key, max_retries=2))
