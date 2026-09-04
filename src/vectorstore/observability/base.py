"""Dependency-free observability seam for retrieval.

The core library never imports an observability framework. It emits one
:class:`~vectorstore.hybrid.retriever.RetrievalResult` per request to a
:class:`RetrievalTraceObserver`; the optional
:class:`~vectorstore.observability.otel.OTelRetrievalObserver` reconstructs
spans from the result's phase timings, and
applications can bridge to whatever exporter stack they already run.

The result deliberately does not retain query text. It does contain hydrated
hit chunks for application use, so custom observers must choose content-safe
fields. The bundled OpenTelemetry observer emits only IDs, ranks, reasons,
counts, and timings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vectorstore.hybrid.retriever import RetrievalResult


@runtime_checkable
class RetrievalTraceObserver(Protocol):
    """Receive one event per completed retrieval request.

    Implementations must not raise: the retrieval facade swallows observer
    exceptions so telemetry can never break retrieval, but a raising
    observer still loses its own events.
    """

    def on_retrieve(self, result: RetrievalResult) -> None:
        """Handle a completed retrieval, successful or degraded."""


class NoOpRetrievalObserver:
    """The default observer: discards every event."""

    def on_retrieve(self, result: RetrievalResult) -> None:
        """Discard a completed retrieval event."""
        return None
