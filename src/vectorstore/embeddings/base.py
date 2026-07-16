"""Embedding provider interface."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Turn text into fixed-width numeric vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The number of elements in each produced embedding."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts while preserving their input order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Providers with asymmetric query/document embedding modes can override
        this method.
        """

        return self.embed_texts([text])[0]
