"""Document catalogs: structured find, lexical search, and retrieval ledgers."""

from .base import (
    CatalogChunk,
    CatalogDocument,
    DocumentCatalog,
    EmbeddingState,
    LexicalUnavailableError,
    RankedHit,
    RetrievalScope,
)
from .sqlite_catalog import SqliteDocumentCatalog

__all__ = [
    "CatalogChunk",
    "CatalogDocument",
    "DocumentCatalog",
    "EmbeddingState",
    "LexicalUnavailableError",
    "RankedHit",
    "RetrievalScope",
    "SqliteDocumentCatalog",
]
