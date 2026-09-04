"""Retrieval observability with a dependency-free core and optional OTel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import NoOpRetrievalObserver, RetrievalTraceObserver

if TYPE_CHECKING:
    from .otel import OpenTelemetryRetrievalObserver, OTelRetrievalObserver


def __getattr__(name: str) -> Any:
    """Load the optional observer without creating a core import cycle."""
    if name in {"OTelRetrievalObserver", "OpenTelemetryRetrievalObserver"}:
        from .otel import OTelRetrievalObserver

        return OTelRetrievalObserver
    raise AttributeError(name)


__all__ = [
    "NoOpRetrievalObserver",
    "OTelRetrievalObserver",
    "OpenTelemetryRetrievalObserver",
    "RetrievalTraceObserver",
]
