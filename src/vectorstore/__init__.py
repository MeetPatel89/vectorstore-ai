"""Extensible vector storage and semantic search."""

from .embeddings import EmbeddingProvider, EmbeddingSpec, OpenAIEmbedding
from .index import VectorIndex
from .models import Chunk, MetadataFilter, MetadataValue, SearchResult
from .records import Record, content_hash, semantic_projection
from .stores import (
    AzureSqlVectorStore,
    ChromaVectorStore,
    FaissVectorStore,
    NumpyVectorStore,
    VectorStore,
    create_store,
    register_store,
)

__all__ = [
    "AzureSqlVectorStore",
    "ChromaVectorStore",
    "Chunk",
    "EmbeddingProvider",
    "EmbeddingSpec",
    "FaissVectorStore",
    "MetadataFilter",
    "MetadataValue",
    "NumpyVectorStore",
    "OpenAIEmbedding",
    "Record",
    "SearchResult",
    "VectorIndex",
    "VectorStore",
    "content_hash",
    "create_store",
    "register_store",
    "semantic_projection",
]
