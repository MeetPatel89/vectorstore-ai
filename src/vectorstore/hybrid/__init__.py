"""Hybrid retrieval: query analysis, RRF fusion, and the Retriever facade."""

from .fusion import DEFAULT_RRF_K, rrf
from .query import QueryAnalyzer, QueryKind, QueryProfile
from .retriever import (
    RetrievalHit,
    RetrievalResult,
    RetrievalTimings,
    Retriever,
    RetrieverConfig,
    merge_scope_filter,
)

__all__ = [
    "DEFAULT_RRF_K",
    "QueryAnalyzer",
    "QueryKind",
    "QueryProfile",
    "RetrievalHit",
    "RetrievalResult",
    "RetrievalTimings",
    "Retriever",
    "RetrieverConfig",
    "merge_scope_filter",
    "rrf",
]
