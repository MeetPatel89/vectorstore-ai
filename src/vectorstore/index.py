"""High-level composition of embedding and storage."""

from vectorstore.embeddings.base import EmbeddingProvider
from vectorstore.models import Chunk, MetadataFilter, SearchResult
from vectorstore.stores.base import VectorStore


class VectorIndex:
    """The primary API for indexing chunks and searching them by text."""

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def index(self, chunks: list[Chunk]) -> None:
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
        if k <= 0:
            return []
        return self.store.search(self.embedder.embed_query(query), k=k, filter=filter)

    def delete(self, ids: list[str]) -> None:
        self.store.delete(ids)

    def count(self) -> int:
        return self.store.count()
