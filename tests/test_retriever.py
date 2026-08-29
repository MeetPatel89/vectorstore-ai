"""Unit and integration tests for the hybrid Retriever facade.

Uses the real SQLite catalog (FTS5) and real NumPy vector stores with the
deterministic FakeEmbedding, so hybrid behavior is exercised end to end
without any network access.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Never, override

import pytest
from conftest import FakeEmbedding

from vectorstore import (
    CatalogChunk,
    CatalogDocument,
    Chunk,
    EmbeddingPrice,
    EmbeddingPricing,
    EmbeddingResult,
    EmbeddingRouter,
    EmbeddingUsage,
    InMemoryBudgetLedger,
    LexicalUnavailableError,
    NumpyVectorStore,
    QueryKind,
    RetrievalResult,
    RetrievalScope,
    RetrievalTraceObserver,
    Retriever,
    RetrieverConfig,
    SqliteDocumentCatalog,
    UsageStatus,
    VectorStore,
    build_retriever,
)
from vectorstore.hybrid.retriever import merge_scope_filter
from vectorstore.models import MetadataFilter, MetadataValue


class FailingEmbedding(FakeEmbedding):
    """A provider whose every call raises, for failure-path tests."""

    @override
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider exploded")

    @override
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("provider exploded")


class UsageEmbedding(FakeEmbedding):
    """A provider that reports authoritative usage with query vectors."""

    @override
    def embed_query_with_usage(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[self.embed_query(text)],
            usage=EmbeddingUsage(total_tokens=321),
        )


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[RetrievalResult] = []

    def on_retrieve(self, result: RetrievalResult) -> None:
        self.events.append(result)


class RaisingObserver:
    def on_retrieve(self, result: RetrievalResult) -> None:
        raise RuntimeError("observer exploded")


DOCUMENTS = [
    CatalogDocument(
        doc_id="doc-payments",
        title="Payment reporting data missing",
        doc_type="incident",
        tenant_id="acme",
        visibility="internal",
    ),
    CatalogDocument(
        doc_id="doc-network",
        title="Network resets in checkout",
        doc_type="incident",
        tenant_id="acme",
        visibility="internal",
    ),
    CatalogDocument(
        doc_id="doc-secret",
        title="Restricted runbook",
        doc_type="runbook",
        tenant_id="other-tenant",
        visibility="restricted",
    ),
]

CHUNKS = [
    CatalogChunk(
        chunk_id="chunk-payments",
        doc_id="doc-payments",
        text=(
            "Incident INC-1104: payment reconciliation reports are missing "
            "rows after the nightly ETL job."
        ),
    ),
    CatalogChunk(
        chunk_id="chunk-network",
        doc_id="doc-network",
        text=(
            "Customers see ERR_CONNECTION_RESET during checkout when the "
            "load balancer restarts."
        ),
    ),
    CatalogChunk(
        chunk_id="chunk-secret",
        doc_id="doc-secret",
        text="Restricted procedure for rotating payment signing keys.",
    ),
]

CHUNK_METADATA: dict[str, dict[str, MetadataValue]] = {
    "chunk-payments": {"tenant_id": "acme", "visibility": "internal"},
    "chunk-network": {"tenant_id": "acme", "visibility": "internal"},
    "chunk-secret": {"tenant_id": "other-tenant", "visibility": "restricted"},
}


@pytest.fixture
def catalog() -> SqliteDocumentCatalog:
    catalog = SqliteDocumentCatalog()
    catalog.upsert_documents(DOCUMENTS)
    catalog.upsert_chunks(CHUNKS)
    return catalog


def make_store(embedder: FakeEmbedding) -> NumpyVectorStore:
    store = NumpyVectorStore()
    chunks = [
        Chunk(
            id=chunk.chunk_id, text=chunk.text, metadata=CHUNK_METADATA[chunk.chunk_id]
        )
        for chunk in CHUNKS
    ]
    store.upsert(chunks, embedder.embed_texts([chunk.text for chunk in chunks]))
    return store


@pytest.fixture
def primary() -> FakeEmbedding:
    return FakeEmbedding(dimension=64, provider="openai-fake", model="primary-model")


@pytest.fixture
def fallback() -> FakeEmbedding:
    return FakeEmbedding(dimension=32, provider="local-fake", model="fallback-model")


def make_retriever(
    catalog: SqliteDocumentCatalog,
    primary: FakeEmbedding,
    fallback: FakeEmbedding | None = None,
    *,
    stores: Mapping[str, VectorStore] | None = None,
    observer: RetrievalTraceObserver | None = None,
    config: RetrieverConfig | None = None,
    primary_enabled: bool = True,
) -> Retriever:
    router = EmbeddingRouter(primary, fallback, primary_enabled=primary_enabled)
    if stores is None:
        stores = {primary.spec.space_id: make_store(primary)}
        if fallback is not None:
            stores[fallback.spec.space_id] = make_store(fallback)
    return Retriever(
        catalog=catalog,
        stores=stores,
        router=router,
        observer=observer,
        config=config,
    )


class TestHybridRetrieve:
    def test_hybrid_returns_relevant_chunk_with_provenance(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        retriever = make_retriever(catalog, primary)
        result = retriever.retrieve("payment reconciliation reports missing")

        assert result.hits
        top = result.hits[0]
        assert top.chunk.chunk_id == "chunk-payments"
        assert top.dense_rank is not None
        assert top.lexical_rank is not None
        assert result.provider == "openai-fake"
        assert result.provider_reason == "primary"
        assert result.fallback_occurred is False
        assert result.degraded is False
        assert result.query_kind is QueryKind.NATURAL
        assert result.dense_candidates > 0
        assert result.lexical_candidates > 0
        assert result.timings.total_ms > 0
        assert result.errors == ()

    def test_identifier_query_upweights_lexical(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        retriever = make_retriever(catalog, primary)
        result = retriever.retrieve("INC-1104")
        assert result.query_kind is QueryKind.IDENTIFIER
        assert result.lexical_weight == 2.0
        assert result.hits[0].chunk.chunk_id == "chunk-payments"

    def test_k_override_truncates_results(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        retriever = make_retriever(catalog, primary)
        result = retriever.retrieve("payment checkout reports", k=1)
        assert len(result.hits) == 1

    def test_empty_query_rejected(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        retriever = make_retriever(catalog, primary)
        with pytest.raises(ValueError, match="find\\(\\)"):
            retriever.retrieve("   ")

    def test_structured_find_short_circuit(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        retriever = make_retriever(catalog, primary)
        docs = retriever.find(filter={"doc_type": "incident"})
        assert {doc.doc_id for doc in docs} == {"doc-payments", "doc-network"}

    def test_primary_api_usage_is_recorded_in_budget_ledger(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        primary = UsageEmbedding(
            dimension=64,
            provider="openai-fake",
            model="primary-model",
        )
        ledger = InMemoryBudgetLedger()
        router = EmbeddingRouter(
            primary,
            ledger=ledger,
            daily_budget_usd="1.00",
            cost_per_million_tokens="0.02",
        )
        retriever = Retriever(
            catalog=catalog,
            stores={primary.spec.space_id: make_store(primary)},
            router=router,
        )

        result = retriever.retrieve("payment reconciliation")

        assert result.provider == "openai-fake"
        assert ledger.tokens_today("openai-fake") == 321
        record = ledger.usage_records()[0]
        assert record.status is UsageStatus.COMMITTED
        assert record.charge.tokens == 321
        assert record.charge.model == "primary-model"

    def test_filter_pushdown_restricts_both_branches(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        retriever = make_retriever(catalog, primary)
        result = retriever.retrieve(
            "payment signing keys", filter={"visibility": "restricted"}
        )
        assert {hit.chunk.chunk_id for hit in result.hits} <= {"chunk-secret"}


class TestScopeEnforcement:
    def test_scope_excludes_other_tenants_everywhere(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        retriever = make_retriever(catalog, primary)
        scope = RetrievalScope(tenant_id="acme", visibility=("internal",))
        result = retriever.retrieve("payment signing keys rotation", scope=scope)
        returned = {hit.chunk.chunk_id for hit in result.hits}
        assert "chunk-secret" not in returned

    def test_merge_scope_filter_is_conservative(self) -> None:
        merged = merge_scope_filter(
            {"doc_type": "incident"},
            RetrievalScope(tenant_id="acme", visibility=("internal", "public")),
        )
        assert merged == {
            "doc_type": "incident",
            "tenant_id": "acme",
            "visibility": {"$in": ["internal", "public"]},
        }

    def test_merge_scope_filter_passthrough_without_scope(self) -> None:
        original: MetadataFilter = {"doc_type": "incident"}
        assert merge_scope_filter(original, None) is original
        assert merge_scope_filter(original, RetrievalScope()) is original


class TestDegradation:
    def test_primary_failure_falls_forward_to_fallback_space(
        self,
        catalog: SqliteDocumentCatalog,
        fallback: FakeEmbedding,
    ) -> None:
        broken_primary = FailingEmbedding(
            dimension=64, provider="openai-fake", model="primary-model"
        )
        stores = {
            broken_primary.spec.space_id: NumpyVectorStore(),
            fallback.spec.space_id: make_store(fallback),
        }
        retriever = make_retriever(catalog, broken_primary, fallback, stores=stores)
        result = retriever.retrieve("payment reconciliation reports")

        assert result.provider == "local-fake"
        assert result.fallback_occurred is True
        assert result.degraded is False
        assert any("embedding failed" in message for message in result.errors)
        assert result.hits

    def test_primary_failure_releases_budget_reservation(
        self,
        catalog: SqliteDocumentCatalog,
        fallback: FakeEmbedding,
    ) -> None:
        broken_primary = FailingEmbedding(
            dimension=64,
            provider="openai-fake",
            model="primary-model",
        )
        ledger = InMemoryBudgetLedger()
        router = EmbeddingRouter(
            broken_primary,
            fallback,
            ledger=ledger,
            daily_budget_usd="1.00",
            cost_per_million_tokens="1.00",
        )
        retriever = Retriever(
            catalog=catalog,
            stores={
                broken_primary.spec.space_id: NumpyVectorStore(),
                fallback.spec.space_id: make_store(fallback),
            },
            router=router,
        )

        result = retriever.retrieve("payment reconciliation reports")

        assert result.provider == "local-fake"
        assert ledger.spent_today_nanos() == 0
        assert ledger.usage_records()[0].status is UsageStatus.RELEASED

    def test_router_selected_fallback_is_not_marked_degraded(
        self,
        catalog: SqliteDocumentCatalog,
        primary: FakeEmbedding,
        fallback: FakeEmbedding,
    ) -> None:
        retriever = make_retriever(catalog, primary, fallback, primary_enabled=False)
        result = retriever.retrieve("payment reconciliation reports")
        assert result.provider == "local-fake"
        assert result.provider_reason == "openai_disabled"
        assert result.fallback_occurred is True
        assert result.degraded is False

    def test_both_providers_failing_degrades_to_lexical(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        broken_primary = FailingEmbedding(
            dimension=64, provider="openai-fake", model="primary-model"
        )
        broken_fallback = FailingEmbedding(
            dimension=32, provider="local-fake", model="fallback-model"
        )
        stores = {
            broken_primary.spec.space_id: NumpyVectorStore(),
            broken_fallback.spec.space_id: NumpyVectorStore(),
        }
        retriever = make_retriever(
            catalog, broken_primary, broken_fallback, stores=stores
        )
        result = retriever.retrieve("payment reconciliation reports")

        assert result.degraded is True
        assert result.provider is None
        assert result.dense_candidates == 0
        # Lexical still found the payments chunk.
        assert result.hits[0].chunk.chunk_id == "chunk-payments"

    def test_missing_store_for_selected_space_degrades_gracefully(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        retriever = make_retriever(
            catalog, primary, stores={"wrong_space": NumpyVectorStore()}
        )
        result = retriever.retrieve("payment reconciliation reports")
        assert result.degraded is True
        assert any("no vector store" in message for message in result.errors)
        assert result.hits  # lexical carried the request

    def test_lexical_unavailable_degrades_to_dense(
        self, primary: FakeEmbedding
    ) -> None:
        class NoFtsCatalog(SqliteDocumentCatalog):
            @override
            def search_lexical(self, *args: object, **kwargs: object) -> Never:
                raise LexicalUnavailableError("FTS5 not available")

        catalog = NoFtsCatalog()
        catalog.upsert_documents(DOCUMENTS)
        catalog.upsert_chunks(CHUNKS)
        retriever = make_retriever(catalog, primary)
        result = retriever.retrieve("payment reconciliation reports")

        assert result.degraded is True
        assert any("lexical unavailable" in message for message in result.errors)
        assert result.lexical_candidates == 0
        assert result.hits  # dense carried the request

    def test_dense_disabled_config_is_lexical_only(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        retriever = make_retriever(
            catalog, primary, config=RetrieverConfig(dense_enabled=False)
        )
        result = retriever.retrieve("payment reconciliation")
        assert result.provider is None
        assert result.degraded is False
        assert result.dense_candidates == 0
        assert result.hits

    def test_chunks_unknown_to_catalog_are_dropped(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        store = make_store(primary)
        store.upsert(
            [Chunk(id="chunk-ghost", text="payment reconciliation ghost", metadata={})],
            primary.embed_texts(["payment reconciliation ghost"]),
        )
        retriever = make_retriever(
            catalog, primary, stores={primary.spec.space_id: store}
        )
        result = retriever.retrieve("payment reconciliation ghost")
        assert all(hit.chunk.chunk_id != "chunk-ghost" for hit in result.hits)


class TestSpaceSafety:
    def test_store_dimension_mismatch_fails_fast(
        self,
        catalog: SqliteDocumentCatalog,
        primary: FakeEmbedding,
        fallback: FakeEmbedding,
    ) -> None:
        wrong_store = make_store(fallback)  # 32-dim store
        with pytest.raises(ValueError, match="one store per embedding space"):
            make_retriever(
                catalog,
                primary,
                stores={primary.spec.space_id: wrong_store},
            )


class TestObserver:
    def test_observer_receives_one_event_per_request(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        observer = RecordingObserver()
        retriever = make_retriever(catalog, primary, observer=observer)
        retriever.retrieve("payment reconciliation")
        retriever.retrieve("INC-1104")

        assert len(observer.events) == 2
        assert observer.events[0].query_id != observer.events[1].query_id
        assert observer.events[1].query_kind is QueryKind.IDENTIFIER

    def test_raising_observer_never_breaks_retrieval(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        retriever = make_retriever(catalog, primary, observer=RaisingObserver())
        result = retriever.retrieve("payment reconciliation")
        assert result.hits


class TestBuildRetriever:
    def test_builds_hybrid_retriever_with_catalog_ledger(
        self,
        catalog: SqliteDocumentCatalog,
        primary: FakeEmbedding,
        fallback: FakeEmbedding,
    ) -> None:
        retriever = build_retriever(
            catalog,
            primary=primary,
            primary_store=make_store(primary),
            fallback=fallback,
            fallback_store=make_store(fallback),
            daily_budget_usd=5.0,
            cost_per_million_tokens=0.02,
        )
        result = retriever.retrieve("payment reconciliation")
        assert result.provider == "openai-fake"
        assert result.hits

    def test_forwards_custom_pricing_to_composed_router(
        self,
        catalog: SqliteDocumentCatalog,
        primary: FakeEmbedding,
    ) -> None:
        pricing = EmbeddingPricing(
            (
                EmbeddingPrice.from_usd_per_million(
                    primary.spec.provider,
                    primary.spec.model,
                    "0.50",
                    version="retriever-contract-v1",
                ),
            )
        )
        retriever = build_retriever(
            catalog,
            primary=primary,
            primary_store=make_store(primary),
            daily_budget_usd="1.00",
            pricing=pricing,
        )

        result = retriever.retrieve("payment reconciliation")

        assert result.provider == "openai-fake"
        record = catalog.usage_records()[0]
        assert record.charge.price_version == "retriever-contract-v1"

    def test_lexical_only_when_no_providers(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        retriever = build_retriever(catalog)
        assert retriever.config.dense_enabled is False
        result = retriever.retrieve("payment reconciliation")
        assert result.provider is None
        assert result.degraded is False
        assert result.hits

    def test_router_and_providers_are_mutually_exclusive(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        router = EmbeddingRouter(primary)
        with pytest.raises(ValueError, match="not both"):
            build_retriever(catalog, router=router, primary=primary)

    def test_fallback_store_without_fallback_rejected(
        self, catalog: SqliteDocumentCatalog, primary: FakeEmbedding
    ) -> None:
        with pytest.raises(ValueError, match="fallback_store"):
            build_retriever(
                catalog,
                primary=primary,
                primary_store=make_store(primary),
                fallback_store=NumpyVectorStore(),
            )

    def test_store_dimension_validated_against_provider(
        self,
        catalog: SqliteDocumentCatalog,
        primary: FakeEmbedding,
        fallback: FakeEmbedding,
    ) -> None:
        with pytest.raises(ValueError, match="primary store"):
            build_retriever(
                catalog,
                primary=primary,
                primary_store=make_store(fallback),
            )


class TestConfigValidation:
    def test_non_positive_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="final_top_k"):
            RetrieverConfig(final_top_k=0)
        with pytest.raises(ValueError, match="rrf_k"):
            RetrieverConfig(rrf_k=-1)
