"""Vector store interface."""

from abc import ABC, abstractmethod

from vectorstore.models import Chunk, MetadataFilter, SearchResult


class VectorStore(ABC):
    """Store embeddings and their chunks, and perform nearest-neighbor search.

    A store instance holds vectors from exactly one embedding space; callers
    must never mix vectors produced by different embedding models in the same
    store.
    """

    @property
    def dimension(self) -> int | None:
        """The vector width this store requires, if one is established.

        Backends that know their dimension (from configuration, prior
        upserts, or persisted data) override this so composition layers can
        reject embedding providers from an incompatible embedding space
        before any vectors are written.
        """
        return None

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
