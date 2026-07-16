"""Extensible vector storage and semantic search."""

from .embeddings import EmbeddingProvider, OpenAIEmbedding
from .index import VectorIndex
from .models import Chunk, MetadataFilter, MetadataValue, SearchResult
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
    "FaissVectorStore",
    "MetadataFilter",
    "MetadataValue",
    "NumpyVectorStore",
    "OpenAIEmbedding",
    "SearchResult",
    "VectorIndex",
    "VectorStore",
    "create_store",
    "register_store",
]
