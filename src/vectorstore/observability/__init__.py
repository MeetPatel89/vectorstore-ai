"""Observability seam: dependency-free observer protocol and no-op default."""

from .base import NoOpRetrievalObserver, RetrievalTraceObserver

__all__ = ["NoOpRetrievalObserver", "RetrievalTraceObserver"]
