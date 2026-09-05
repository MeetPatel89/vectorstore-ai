from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from vectorstore import (
    CatalogChunk,
    OpenTelemetryRetrievalObserver,
    OTelRetrievalObserver,
    QueryKind,
    RetrievalHit,
    RetrievalResult,
    RetrievalTimings,
    RetrievalTraceObserver,
)
from vectorstore.observability import OTelRetrievalObserver as LazyOTelObserver


def _result(*, degraded: bool = False) -> RetrievalResult:
    return RetrievalResult(
        hits=(
            RetrievalHit(
                chunk=CatalogChunk(
                    "chunk-1",
                    "doc-1",
                    "secret document text that telemetry must never capture",
                ),
                score=0.03,
                dense_rank=1,
                lexical_rank=2,
            ),
        ),
        query_id="query-123",
        query_kind=QueryKind.IDENTIFIER,
        provider="local",
        model="all-MiniLM-L6-v2",
        space_id="local__minilm__384__v1",
        provider_reason="budget_daily_exceeded",
        fallback_occurred=True,
        degraded=degraded,
        dense_candidates=7,
        lexical_candidates=5,
        dense_weight=1.0,
        lexical_weight=2.0,
        timings=RetrievalTimings(
            dense_ms=6.0,
            lexical_ms=4.0,
            total_ms=20.0,
            embedding_ms=2.0,
            dense_search_ms=3.0,
            fusion_ms=1.0,
        ),
        errors=(("lexical unavailable",) if degraded else ()),
        filters_count=2,
        scope_tenant="acme",
        dense_top_k=50,
        lexical_top_k=40,
        final_top_k=10,
        embedding_input_tokens=12,
        embedding_estimated_usd=0.000001,
        dense_backend="numpy",
        lexical_backend="azure-sql",
    )


def _observer() -> tuple[OTelRetrievalObserver, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("vectorstore.tests")
    observer = OTelRetrievalObserver(
        tracer,
        attributes={"langfuse.trace.metadata.environment": "test"},
        time_ns=lambda: 2_000_000_000,
    )
    return observer, exporter


def test_optional_observer_satisfies_protocol_and_alias() -> None:
    observer, _ = _observer()

    assert isinstance(observer, RetrievalTraceObserver)
    assert OpenTelemetryRetrievalObserver is OTelRetrievalObserver
    assert LazyOTelObserver is OTelRetrievalObserver


def test_emits_root_and_timed_phase_children_with_semantic_attributes() -> None:
    observer, exporter = _observer()

    observer.on_retrieve(_result())

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    assert set(by_name) == {
        "retrieve",
        "embeddings all-MiniLM-L6-v2",
        "dense.search numpy",
        "lexical.search azure-sql",
        "fuse rrf",
    }
    root = by_name["retrieve"]
    root_context = root.get_span_context()
    assert root_context is not None
    assert all(
        span.parent is not None and span.parent.span_id == root_context.span_id
        for name, span in by_name.items()
        if name != "retrieve"
    )
    assert root.start_time == 1_980_000_000
    assert root.end_time == 2_000_000_000

    attributes = root.attributes
    assert attributes is not None
    assert attributes["retrieval.query_id"] == "query-123"
    assert attributes["retrieval.query.kind"] == "identifier"
    assert attributes["retrieval.filters.count"] == 2
    assert attributes["retrieval.scope.tenant"] == "acme"
    assert attributes["retrieval.provider.selected"] == "local"
    assert attributes["retrieval.provider.reason"] == "budget_daily_exceeded"
    assert attributes["retrieval.fallback"] is True
    assert attributes["retrieval.result.ids"] == ("chunk-1",)
    assert attributes["retrieval.embedding.estimated_usd"] == 0.000001
    assert attributes["langfuse.trace.metadata.environment"] == "test"

    embedding = by_name["embeddings all-MiniLM-L6-v2"]
    embedding_attributes = embedding.attributes
    assert embedding_attributes is not None
    assert embedding_attributes["gen_ai.operation.name"] == "embeddings"
    assert embedding_attributes["gen_ai.request.model"] == "all-MiniLM-L6-v2"
    assert embedding_attributes["gen_ai.usage.input_tokens"] == 12
    assert embedding.start_time is not None
    assert embedding.end_time is not None
    assert embedding.end_time - embedding.start_time == 2_000_000


def test_emits_fallback_budget_and_degraded_events_and_error_status() -> None:
    observer, exporter = _observer()

    observer.on_retrieve(_result(degraded=True))

    root = next(
        span for span in exporter.get_finished_spans() if span.name == "retrieve"
    )
    assert [event.name for event in root.events] == [
        "retrieval.fallback",
        "retrieval.degraded",
        "budget.threshold_crossed",
    ]
    assert root.status.status_code is StatusCode.ERROR
    assert root.attributes is not None
    assert root.attributes["error.type"] == "retrieval.degraded"


def test_observer_never_captures_query_or_chunk_text() -> None:
    observer, exporter = _observer()

    observer.on_retrieve(_result())

    representation = repr(exporter.get_finished_spans())
    assert "secret document text" not in representation
    assert "query text" not in representation


class RaisingTracer:
    def start_as_current_span(self, *args: object, **kwargs: object) -> Any:
        raise RuntimeError("tracer failed")

    def start_span(self, *args: object, **kwargs: object) -> Any:
        raise RuntimeError("tracer failed")


def test_direct_observer_call_is_fail_open() -> None:
    observer = OTelRetrievalObserver(RaisingTracer())

    observer.on_retrieve(_result())


@pytest.mark.parametrize("value", [None, object()])
def test_rejects_non_tracer_objects(value: object) -> None:
    with pytest.raises(TypeError, match="tracer"):
        OTelRetrievalObserver(value)
