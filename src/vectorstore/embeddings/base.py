"""Embedding provider interface and embedding-space identity."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

_SANITIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def _sanitize_space_part(value: str) -> str:
    sanitized = _SANITIZE_PATTERN.sub("_", value.lower()).strip("_")
    if not sanitized:
        raise ValueError(f"cannot derive a space identifier from {value!r}")
    return sanitized


@dataclass(frozen=True)
class EmbeddingSpec:
    """The identity of an embedding space.

    Vectors are only comparable within a single space: the same provider,
    model, dimensionality, and embedding configuration version. The
    ``space_id`` derives stable, filesystem- and SQL-identifier-safe names
    for per-space vector indexes, so vectors from different spaces can never
    be mixed in one index.
    """

    provider: str
    model: str
    dimension: int
    version: str = "v1"

    def __post_init__(self) -> None:
        for label in ("provider", "model", "version"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        if (
            not isinstance(self.dimension, int)
            or isinstance(self.dimension, bool)
            or self.dimension <= 0
        ):
            raise ValueError("dimension must be a positive integer")

    @property
    def space_id(self) -> str:
        """A stable identifier safe for directories, tables, and collections."""
        return "__".join(
            _sanitize_space_part(part)
            for part in (
                self.provider,
                self.model,
                str(self.dimension),
                self.version,
            )
        )


class EmbeddingProvider(ABC):
    """Turn text into fixed-width numeric vectors."""

    @property
    @abstractmethod
    def spec(self) -> EmbeddingSpec:
        """The embedding space this provider's vectors belong to."""

    @property
    def dimension(self) -> int:
        """The number of elements in each produced embedding."""
        return self.spec.dimension

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts while preserving their input order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Providers with asymmetric query/document embedding modes can override
        this method.
        """
        return self.embed_texts([text])[0]
