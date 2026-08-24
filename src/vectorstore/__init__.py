"""Extensible vector storage and semantic search."""

from .catalog import (
    CatalogChunk,
    CatalogDocument,
    DocumentCatalog,
    EmbeddingState,
    LexicalUnavailableError,
    RankedHit,
    RetrievalScope,
    SqliteDocumentCatalog,
)
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
    "CatalogChunk",
    "CatalogDocument",
    "ChromaVectorStore",
    "Chunk",
    "CircuitBreaker",
    "DocumentCatalog",
    "EmbeddingProvider",
    "EmbeddingRouter",
    "EmbeddingSpec",
    "EmbeddingState",
    "FaissVectorStore",
    "InMemoryBudgetLedger",
    "LexicalUnavailableError",
    "MetadataFilter",
    "MetadataValue",
    "NoProviderAvailableError",
    "NumpyVectorStore",
    "OpenAIEmbedding",
    "ProviderSelection",
    "RankedHit",
    "Record",
    "RetrievalScope",
    "SelectionReason",
    "SentenceTransformerEmbedding",
    "SearchResult",
    "SqliteDocumentCatalog",
    "VectorIndex",
    "VectorStore",
    "content_hash",
    "create_store",
    "register_store",
    "semantic_projection",
]
