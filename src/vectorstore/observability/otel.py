"""OpenTelemetry retrieval observer behind the optional ``otel`` extra.

The observer receives the dependency-free :class:`RetrievalResult` emitted by
the core facade and reconstructs a root retrieval span plus phase children.
Applications inject their own tracer and therefore retain ownership of tracer
providers, processors, exporters, sampling, and resource configuration.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from vectorstore.hybrid.retriever import RetrievalResult

AttributeValue = str | bool | int | float | tuple[str, ...] | tuple[float, ...]


class OTelRetrievalObserver:
    """Emit OpenTelemetry spans from completed, content-safe retrieval results.

    Parameters
    ----------
    tracer:
        An application-configured OpenTelemetry ``Tracer``. The observer never
        creates a provider or exporter.
    attributes:
        Optional root-span attributes, including vendor-specific keys such as
        ``langfuse.trace.metadata.*``. Generated retrieval attributes take
        precedence when a key collides.

    Notes
    -----
    Query text is not retained on ``RetrievalResult``. Hydrated result chunks
    do contain document text for application use, but this observer never reads
    or emits it. Query IDs, tenant IDs, and result IDs are emitted because the
    architecture explicitly treats those as safe operational identifiers.
    """

    def __init__(
        self,
        tracer: Any,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not callable(getattr(tracer, "start_as_current_span", None)):
            raise TypeError("tracer must provide start_as_current_span()")
        if not callable(getattr(tracer, "start_span", None)):
            raise TypeError("tracer must provide start_span()")
        if not callable(time_ns):
            raise TypeError("time_ns must be callable")
        custom_attributes = dict(attributes or {})
        if any(not isinstance(key, str) or not key for key in custom_attributes):
            raise ValueError("OpenTelemetry attribute keys must be non-empty strings")

        try:
            from opentelemetry.trace import SpanKind, Status, StatusCode
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OTelRetrievalObserver requires the otel extra; install it with "
                "uv sync --extra otel"
            ) from exc

        self._tracer = tracer
        self._attributes = custom_attributes
        self._time_ns = time_ns
        self._span_kind_internal = SpanKind.INTERNAL
        self._span_kind_client = SpanKind.CLIENT
        self._status = Status
        self._status_code_error = StatusCode.ERROR

    def on_retrieve(self, result: RetrievalResult) -> None:
        """Create one trace for a completed retrieval without raising."""
        try:
            self._emit(result)
        except Exception:
            # Observability is explicitly fail-open. The Retriever also guards
            # the observer boundary, but keeping the implementation safe makes
            # direct calls honor the same contract.
            return

    def _emit(self, result: RetrievalResult) -> None:
        ended_ns = self._time_ns()
        started_ns = max(0, ended_ns - _milliseconds_to_ns(result.timings.total_ms))
        attributes = {**self._attributes, **_root_attributes(result)}

        with self._tracer.start_as_current_span(
            "retrieve",
            kind=self._span_kind_internal,
            attributes=attributes,
            start_time=started_ns,
            end_on_exit=False,
        ) as root:
            try:
                self._emit_phases(result, started_ns, ended_ns)
                self._emit_events(root, result)
                if result.degraded and result.errors:
                    root.set_status(
                        self._status(
                            self._status_code_error,
                            description=(
                                "one or more retrieval signals were unavailable"
                            ),
                        )
                    )
            finally:
                root.end(end_time=ended_ns)

    def _emit_phases(
        self,
        result: RetrievalResult,
        started_ns: int,
        ended_ns: int,
    ) -> None:
        phase_start = started_ns

        embedding_ms = result.timings.embedding_ms
        if result.model is not None and embedding_ms is not None:
            phase_start = self._emit_child(
                f"embeddings {result.model}",
                kind=self._span_kind_client,
                attributes=_embedding_attributes(result),
                start_ns=phase_start,
                duration_ms=embedding_ms,
                root_end_ns=ended_ns,
            )

        dense_search_ms = result.timings.dense_search_ms
        if (
            dense_search_ms is None
            and embedding_ms is None
            and result.provider is not None
        ):
            dense_search_ms = result.timings.dense_ms
        if dense_search_ms is not None:
            phase_start = self._emit_child(
                f"dense.search {result.dense_backend or 'unknown'}",
                kind=self._span_kind_client,
                attributes={
                    "retrieval.backend": result.dense_backend or "unknown",
                    "retrieval.top_k": result.dense_top_k,
                    "retrieval.result.count": result.dense_candidates,
                },
                start_ns=phase_start,
                duration_ms=dense_search_ms,
                root_end_ns=ended_ns,
            )

        if result.timings.lexical_ms is not None:
            phase_start = self._emit_child(
                f"lexical.search {result.lexical_backend or 'unknown'}",
                kind=self._span_kind_client,
                attributes={
                    "retrieval.backend": result.lexical_backend or "unknown",
                    "retrieval.top_k": result.lexical_top_k,
                    "retrieval.result.count": result.lexical_candidates,
                },
                start_ns=phase_start,
                duration_ms=result.timings.lexical_ms,
                root_end_ns=ended_ns,
            )

        if result.timings.fusion_ms is not None:
            self._emit_child(
                f"fuse {result.fusion_method}",
                kind=self._span_kind_internal,
                attributes={
                    "retrieval.fusion.method": result.fusion_method,
                    "retrieval.fusion.weights": (
                        result.dense_weight,
                        result.lexical_weight,
                    ),
                    "retrieval.result.count": len(result.hits),
                },
                start_ns=phase_start,
                duration_ms=result.timings.fusion_ms,
                root_end_ns=ended_ns,
            )

    def _emit_child(
        self,
        name: str,
        *,
        kind: Any,
        attributes: Mapping[str, AttributeValue],
        start_ns: int,
        duration_ms: float,
        root_end_ns: int,
    ) -> int:
        end_ns = min(root_end_ns, start_ns + _milliseconds_to_ns(duration_ms))
        span = self._tracer.start_span(
            name,
            kind=kind,
            attributes=dict(attributes),
            start_time=start_ns,
        )
        span.end(end_time=end_ns)
        return end_ns

    @staticmethod
    def _emit_events(root: Any, result: RetrievalResult) -> None:
        if result.fallback_occurred:
            root.add_event(
                "retrieval.fallback",
                {
                    "retrieval.provider.selected": result.provider or "none",
                    "retrieval.provider.reason": result.provider_reason or "unknown",
                },
            )
        if result.degraded:
            root.add_event(
                "retrieval.degraded",
                {"retrieval.error.count": len(result.errors)},
            )
        if result.provider_reason in {
            "budget_daily_exceeded",
            "budget_monthly_exceeded",
        }:
            root.add_event(
                "budget.threshold_crossed",
                {"retrieval.provider.reason": result.provider_reason},
            )


def _root_attributes(result: RetrievalResult) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {
        "retrieval.query_id": result.query_id,
        "retrieval.query.kind": str(result.query_kind),
        "retrieval.filters.count": result.filters_count,
        "retrieval.fallback": result.fallback_occurred,
        "retrieval.degraded": result.degraded,
        "retrieval.dense.top_k": result.dense_top_k,
        "retrieval.lexical.top_k": result.lexical_top_k,
        "retrieval.final.top_k": result.final_top_k,
        "retrieval.fusion.method": result.fusion_method,
        "retrieval.fusion.weights": (
            result.dense_weight,
            result.lexical_weight,
        ),
        "retrieval.result.count": len(result.hits),
        "retrieval.result.ids": tuple(hit.chunk.chunk_id for hit in result.hits),
        "retrieval.latency_ms": result.timings.total_ms,
        "retrieval.error.count": len(result.errors),
    }
    _set_if_not_none(attributes, "retrieval.scope.tenant", result.scope_tenant)
    _set_if_not_none(attributes, "retrieval.provider.selected", result.provider)
    _set_if_not_none(attributes, "retrieval.provider.model", result.model)
    _set_if_not_none(attributes, "retrieval.provider.space_id", result.space_id)
    _set_if_not_none(
        attributes,
        "retrieval.provider.reason",
        result.provider_reason,
    )
    _set_if_not_none(attributes, "retrieval.dense.backend", result.dense_backend)
    _set_if_not_none(
        attributes,
        "retrieval.lexical.backend",
        result.lexical_backend,
    )
    _set_if_not_none(
        attributes,
        "retrieval.dense.latency_ms",
        result.timings.dense_ms,
    )
    _set_if_not_none(
        attributes,
        "retrieval.lexical.latency_ms",
        result.timings.lexical_ms,
    )
    _set_if_not_none(
        attributes,
        "retrieval.embedding.estimated_usd",
        result.embedding_estimated_usd,
    )
    if result.errors:
        attributes["error.type"] = (
            "retrieval.degraded" if result.degraded else "retrieval.recovered"
        )
        attributes["retrieval.error.types"] = tuple(
            message.partition(":")[0] for message in result.errors
        )
    return attributes


def _embedding_attributes(result: RetrievalResult) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {
        "gen_ai.operation.name": "embeddings",
        "gen_ai.request.model": result.model or "unknown",
    }
    _set_if_not_none(
        attributes,
        "gen_ai.usage.input_tokens",
        result.embedding_input_tokens,
    )
    _set_if_not_none(attributes, "retrieval.provider.selected", result.provider)
    _set_if_not_none(attributes, "retrieval.provider.space_id", result.space_id)
    return attributes


def _set_if_not_none(
    attributes: dict[str, AttributeValue],
    key: str,
    value: str | float | None,
) -> None:
    if value is not None:
        attributes[key] = value


def _milliseconds_to_ns(value: float) -> int:
    return max(0, round(value * 1_000_000))


OpenTelemetryRetrievalObserver = OTelRetrievalObserver

__all__ = ["OTelRetrievalObserver", "OpenTelemetryRetrievalObserver"]
