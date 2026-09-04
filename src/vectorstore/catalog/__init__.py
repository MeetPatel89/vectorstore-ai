"""Document catalogs: structured find, lexical search, and retrieval ledgers."""

from .azure_sql_catalog import AzureSqlCatalog, AzureSqlDocumentCatalog
from .base import (
    CatalogChunk,
    CatalogDocument,
    DocumentCatalog,
    EmbeddingState,
    LexicalUnavailableError,
    RankedHit,
    RetrievalCatalog,
    RetrievalScope,
)
from .postgres_catalog import PostgresDocumentCatalog
from .sqlite_catalog import SqliteDocumentCatalog

__all__ = [
    "AzureSqlCatalog",
    "AzureSqlDocumentCatalog",
    "CatalogChunk",
    "CatalogDocument",
    "DocumentCatalog",
    "EmbeddingState",
    "LexicalUnavailableError",
    "PostgresDocumentCatalog",
    "RankedHit",
    "RetrievalCatalog",
    "RetrievalScope",
    "SqliteDocumentCatalog",
]
