"""High-level composition of embedding and storage."""

from vectorstore.embeddings.base import EmbeddingProvider, EmbeddingSpec
from vectorstore.models import Chunk, MetadataFilter, SearchResult
from vectorstore.stores.base import VectorStore


class VectorIndex:
    """The primary API for indexing chunks and searching them by text.

    A ``VectorIndex`` is bound to exactly one embedding space: the space of
    its embedding provider. At construction it rejects stores whose
    established dimension does not match the provider, so vectors from
    different embedding models can never be compared against one another.
    """

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        spec = embedder.spec
        store_dimension = store.dimension
        if store_dimension is not None and store_dimension != spec.dimension:
            raise ValueError(
                f"embedding space {spec.space_id!r} produces "
                f"{spec.dimension}-dimensional vectors but the store expects "
                f"{store_dimension}; use one store per embedding space"
            )
        self._embedder = embedder
        self._store = store
        self._spec = spec

    @property
    def embedder(self) -> EmbeddingProvider:
        """The provider bound to this index's immutable embedding space."""
        return self._embedder

    @property
    def store(self) -> VectorStore:
        """The vector store bound to this index."""
        return self._store

    @property
    def spec(self) -> EmbeddingSpec:
        """The embedding space this index reads from and writes to."""
        return self._spec

    def index(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks into this index's embedding space."""
        if not chunks:
            return
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(
                "embedding provider returned a different number of vectors than texts"
            )
        self.store.upsert(chunks, vectors)

    def search(
        self,
        query: str,
        k: int = 5,
        filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Embed *query* and return the highest-scoring matching chunks."""
        if k <= 0:
            return []
        return self.store.search(self.embedder.embed_query(query), k=k, filter=filter)

    def delete(self, ids: list[str]) -> None:
        """Delete chunks with the requested IDs from the store."""
        self.store.delete(ids)

    def count(self) -> int:
        """Return the number of chunks in the store."""
        return self.store.count()
