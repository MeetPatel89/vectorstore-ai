"""Extensible vector storage and semantic search."""

from .embeddings import (
    BudgetLedger,
    CircuitBreaker,
    EmbeddingProvider,
    EmbeddingRouter,
    EmbeddingSpec,
    InMemoryBudgetLedger,
    NoProviderAvailableError,
    OpenAIEmbedding,
    ProviderSelection,
    SelectionReason,
    SentenceTransformerEmbedding,
)
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
    "BudgetLedger",
    "ChromaVectorStore",
    "Chunk",
    "CircuitBreaker",
    "EmbeddingProvider",
    "EmbeddingRouter",
    "EmbeddingSpec",
    "FaissVectorStore",
    "InMemoryBudgetLedger",
    "MetadataFilter",
    "MetadataValue",
    "NoProviderAvailableError",
    "NumpyVectorStore",
    "OpenAIEmbedding",
    "ProviderSelection",
    "Record",
    "SelectionReason",
    "SentenceTransformerEmbedding",
    "SearchResult",
    "VectorIndex",
    "VectorStore",
    "content_hash",
    "create_store",
    "register_store",
    "semantic_projection",
]
