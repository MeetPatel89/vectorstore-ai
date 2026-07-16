"""Vector store interface."""

from abc import ABC, abstractmethod

from vectorstore.models import Chunk, MetadataFilter, SearchResult


class VectorStore(ABC):
    """Store embeddings and their chunks, and perform nearest-neighbor search."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert new records or fully replace records with matching IDs."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete records by ID; unknown IDs are ignored."""

    @abstractmethod
    def search(
        self,
        vector: list[float],
        k: int = 5,
        filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Return up to *k* chunks ordered by descending cosine similarity."""

    @abstractmethod
    def get(self, ids: list[str]) -> list[Chunk]:
        """Get existing chunks in requested ID order."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored chunks."""
