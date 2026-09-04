"""Tests for the SQLite document catalog: structured, lexical, ledgers, scope."""

import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest
from conftest import FakeEmbedding

from vectorstore.catalog import (
    CatalogChunk,
    CatalogDocument,
    DocumentCatalog,
    LexicalUnavailableError,
    RetrievalScope,
    SqliteDocumentCatalog,
)
from vectorstore.catalog.sqlite_catalog import _fts_match_expression
from vectorstore.embeddings.base import EmbeddingSpec
from vectorstore.embeddings.policy import (
    BudgetLedger,
    EmbeddingRouter,
    SelectionReason,
    UsageStatus,
)
from vectorstore.embeddings.pricing import EmbeddingPrice
from vectorstore.records import content_hash


class FakeClock:
    def __init__(self, iso: str = "2026-08-23T12:00:00+00:00") -> None:
        self.moment = datetime.fromisoformat(iso)

    def __call__(self) -> datetime:
        return self.moment

    def advance_days(self, days: int) -> None:
        self.moment = datetime.fromtimestamp(
            self.moment.timestamp() + days * 86400, tz=UTC
        )


INCIDENTS = [
    CatalogDocument(
        doc_id="INC-1104",
        title="Payment reporting data missing",
        doc_type="incident",
        tenant_id="acme",
        visibility="internal",
        status="OPEN",
        attributes={"severity": 3, "service": "payments-reporting"},
    ),
    CatalogDocument(
        doc_id="INC-2001",
        title="Login latency degradation",
        doc_type="incident",
        tenant_id="acme",
        visibility="public",
        status="RESOLVED",
        attributes={"severity": 1, "service": "auth"},
    ),
    CatalogDocument(
        doc_id="KB-77",
        title="How to rotate database credentials",
        doc_type="kb_article",
        tenant_id=None,
        visibility=None,
        status="PUBLISHED",
        attributes={"severity": 0},
    ),
    CatalogDocument(
        doc_id="INC-3000",
        title="Order export failing for tenant globex",
        doc_type="incident",
        tenant_id="globex",
        visibility="internal",
        status="OPEN",
        attributes={"severity": 2, "service": "orders"},
    ),
]

CHUNKS = [
    CatalogChunk(
        chunk_id="INC-1104:0",
        doc_id="INC-1104",
        text=(
            "Incident: INC-1104\nTitle: Payment reporting data missing\n"
            "Description: nightly payment reconciliation reports are empty"
        ),
    ),
    CatalogChunk(
        chunk_id="INC-2001:0",
        doc_id="INC-2001",
        text=(
            "Incident: INC-2001\nTitle: Login latency degradation\n"
            "Description: users report slow sign-in during peak hours"
        ),
    ),
    CatalogChunk(
        chunk_id="KB-77:0",
        doc_id="KB-77",
        text=(
            "Title: How to rotate database credentials\n"
            "Body: rotate credentials quarterly using the vault CLI"
        ),
    ),
    CatalogChunk(
        chunk_id="INC-3000:0",
        doc_id="INC-3000",
        text=(
            "Incident: INC-3000\nTitle: Order export failing\n"
            "Description: SQLSTATE 23505 duplicate key on order export job"
        ),
    ),
]


def _required_hash(chunk: CatalogChunk) -> str:
    assert chunk.content_hash is not None
    return chunk.content_hash


@pytest.fixture
def catalog() -> Iterator[SqliteDocumentCatalog]:
    with SqliteDocumentCatalog() as instance:
        instance.upsert_documents(INCIDENTS)
        instance.upsert_chunks(CHUNKS)
        yield instance


class TestValueObjects:
    def test_document_requires_doc_id(self) -> None:
        with pytest.raises(ValueError):
            CatalogDocument(doc_id="")

    def test_chunk_requires_nonempty_text(self) -> None:
        with pytest.raises(ValueError):
            CatalogChunk(chunk_id="c1", doc_id="d1", text="   ")

    def test_chunk_computes_content_hash_by_default(self) -> None:
        chunk = CatalogChunk(chunk_id="c1", doc_id="d1", text="hello")
        assert chunk.content_hash == content_hash("hello")

    def test_chunk_rejects_invalid_lifecycle_state(self) -> None:
        with pytest.raises(ValueError, match="chunk_index"):
            CatalogChunk(chunk_id="c1", doc_id="d1", text="hello", chunk_index=-1)
        with pytest.raises(ValueError, match="active"):
            CatalogChunk(
                chunk_id="c1",
                doc_id="d1",
                text="hello",
                active=cast(bool, 1),
            )

    def test_document_snapshots_attributes(self) -> None:
        attributes = {"status_code": 3}
        document = CatalogDocument(doc_id="doc", attributes=attributes)

        attributes["status_code"] = 4

        assert document.attributes == {"status_code": 3}

    def test_scope_rejects_empty_visibility_tuple(self) -> None:
        with pytest.raises(ValueError):
            RetrievalScope(visibility=())

    def test_scope_rejects_empty_tenant(self) -> None:
        with pytest.raises(ValueError):
            RetrievalScope(tenant_id="")

    def test_sqlite_catalog_satisfies_protocols(self) -> None:
        with SqliteDocumentCatalog() as instance:
            assert isinstance(instance, DocumentCatalog)
            assert isinstance(instance, BudgetLedger)


class TestStructuredFind:
    def test_find_all(self, catalog: SqliteDocumentCatalog) -> None:
        assert len(catalog.find()) == 4

    def test_filter_on_column(self, catalog: SqliteDocumentCatalog) -> None:
        docs = catalog.find({"doc_type": "incident", "status": "OPEN"})
        assert {doc.doc_id for doc in docs} == {"INC-1104", "INC-3000"}

    def test_filter_on_json_attribute(self, catalog: SqliteDocumentCatalog) -> None:
        docs = catalog.find({"service": "auth"})
        assert [doc.doc_id for doc in docs] == ["INC-2001"]

    def test_comparison_operator_on_json_attribute(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        docs = catalog.find({"severity": {"$gte": 2}})
        assert {doc.doc_id for doc in docs} == {"INC-1104", "INC-3000"}

    def test_in_operator(self, catalog: SqliteDocumentCatalog) -> None:
        docs = catalog.find({"doc_id": {"$in": ["KB-77", "INC-2001"]}})
        assert {doc.doc_id for doc in docs} == {"KB-77", "INC-2001"}

    def test_limit(self, catalog: SqliteDocumentCatalog) -> None:
        assert len(catalog.find(limit=2)) == 2

    def test_invalid_limit_rejected(self, catalog: SqliteDocumentCatalog) -> None:
        with pytest.raises(ValueError):
            catalog.find(limit=0)

    def test_unsupported_operator_rejected(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        with pytest.raises(ValueError):
            catalog.find({"severity": {"$regex": "x"}})

    def test_attributes_round_trip(self, catalog: SqliteDocumentCatalog) -> None:
        (doc,) = catalog.find({"doc_id": "INC-1104"})
        assert doc.attributes == {"severity": 3, "service": "payments-reporting"}
        assert doc.title == "Payment reporting data missing"

    def test_upsert_updates_existing_document(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        catalog.upsert_documents(
            [CatalogDocument(doc_id="INC-1104", title="updated", status="RESOLVED")]
        )
        (doc,) = catalog.find({"doc_id": "INC-1104"})
        assert doc.title == "updated"
        assert doc.status == "RESOLVED"
        assert len(catalog.find()) == 4


class TestScopeEnforcement:
    def test_tenant_scope_includes_shared_documents(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        docs = catalog.find(scope=RetrievalScope(tenant_id="acme"))
        assert {doc.doc_id for doc in docs} == {"INC-1104", "INC-2001", "KB-77"}

    def test_tenant_scope_excludes_other_tenants(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        docs = catalog.find(scope=RetrievalScope(tenant_id="globex"))
        assert {doc.doc_id for doc in docs} == {"INC-3000", "KB-77"}

    def test_visibility_scope(self, catalog: SqliteDocumentCatalog) -> None:
        docs = catalog.find(scope=RetrievalScope(visibility=("public",)))
        assert {doc.doc_id for doc in docs} == {"INC-2001", "KB-77"}

    def test_combined_scope_and_filter(self, catalog: SqliteDocumentCatalog) -> None:
        docs = catalog.find(
            {"doc_type": "incident"},
            scope=RetrievalScope(tenant_id="acme", visibility=("internal",)),
        )
        assert [doc.doc_id for doc in docs] == ["INC-1104"]

    def test_lexical_search_respects_tenant_scope(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        hits = catalog.search_lexical(
            "export failing", scope=RetrievalScope(tenant_id="acme")
        )
        assert all(hit.chunk_id != "INC-3000:0" for hit in hits)

    def test_lexical_search_within_scope_finds_shared(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        hits = catalog.search_lexical(
            "rotate credentials", scope=RetrievalScope(tenant_id="globex")
        )
        assert hits[0].chunk_id == "KB-77:0"


class TestLexicalSearch:
    def test_natural_language_query(self, catalog: SqliteDocumentCatalog) -> None:
        hits = catalog.search_lexical("payment reconciliation reports")
        assert hits[0].chunk_id == "INC-1104:0"

    def test_identifier_query(self, catalog: SqliteDocumentCatalog) -> None:
        hits = catalog.search_lexical("SQLSTATE 23505")
        assert hits[0].chunk_id == "INC-3000:0"

    def test_quoted_phrase_matches_only_exact_sequence(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        hits = catalog.search_lexical('"payment reconciliation"')
        assert [hit.chunk_id for hit in hits] == ["INC-1104:0"]

    def test_ranks_are_one_based_and_scores_descending(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        hits = catalog.search_lexical("incident title description", k=10)
        assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_results(self, catalog: SqliteDocumentCatalog) -> None:
        assert len(catalog.search_lexical("incident", k=1)) == 1

    def test_filter_pushdown(self, catalog: SqliteDocumentCatalog) -> None:
        hits = catalog.search_lexical("incident", filter={"status": "OPEN"})
        assert {hit.chunk_id for hit in hits} == {"INC-1104:0", "INC-3000:0"}

    def test_inactive_chunks_excluded(self, catalog: SqliteDocumentCatalog) -> None:
        catalog.upsert_chunks(
            [
                CatalogChunk(
                    chunk_id="INC-1104:0",
                    doc_id="INC-1104",
                    text=CHUNKS[0].text,
                    active=False,
                )
            ]
        )
        hits = catalog.search_lexical("payment reconciliation reports")
        assert all(hit.chunk_id != "INC-1104:0" for hit in hits)

    def test_updated_chunk_text_is_searchable(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        catalog.upsert_chunks(
            [
                CatalogChunk(
                    chunk_id="INC-2001:0",
                    doc_id="INC-2001",
                    text="Description: ERR_CONNECTION_RESET seen on login page",
                )
            ]
        )
        hits = catalog.search_lexical("ERR_CONNECTION_RESET")
        assert hits[0].chunk_id == "INC-2001:0"
        assert catalog.search_lexical("slow sign-in during peak") == []

    def test_hostile_query_does_not_raise_syntax_error(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        assert catalog.search_lexical('AND OR NOT ( " * unbalanced') == []

    def test_empty_query_rejected(self, catalog: SqliteDocumentCatalog) -> None:
        with pytest.raises(ValueError):
            catalog.search_lexical("   ")

    def test_invalid_k_rejected(self, catalog: SqliteDocumentCatalog) -> None:
        with pytest.raises(ValueError):
            catalog.search_lexical("incident", k=0)


class TestFtsMatchExpression:
    def test_tokens_are_quoted(self) -> None:
        assert _fts_match_expression("INC-1104 open") == '"INC-1104" "open"'

    def test_quoted_phrase_preserved(self) -> None:
        assert (
            _fts_match_expression('fix "payment reconciliation" now')
            == '"fix" "payment reconciliation" "now"'
        )

    def test_embedded_quotes_escaped(self) -> None:
        assert _fts_match_expression('say"hi') == '"say""hi"'


class TestChunksAndDeletion:
    def test_get_chunks_preserves_requested_order(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        chunks = catalog.get_chunks(["KB-77:0", "INC-1104:0", "missing"])
        assert [chunk.chunk_id for chunk in chunks] == ["KB-77:0", "INC-1104:0"]
        assert chunks[1].text == CHUNKS[0].text

    def test_get_chunks_empty_input(self, catalog: SqliteDocumentCatalog) -> None:
        assert catalog.get_chunks([]) == []

    def test_delete_documents_cascades(self, catalog: SqliteDocumentCatalog) -> None:
        spec = EmbeddingSpec(provider="fake", model="m", dimension=8)
        catalog.mark_embedded("INC-1104:0", spec, _required_hash(CHUNKS[0]))
        catalog.delete_documents(["INC-1104"])
        assert catalog.find({"doc_id": "INC-1104"}) == []
        assert catalog.get_chunks(["INC-1104:0"]) == []
        assert catalog.embedding_state(spec.space_id) == {}
        hits = catalog.search_lexical("payment reconciliation reports")
        assert all(hit.chunk_id != "INC-1104:0" for hit in hits)

    def test_replace_chunks_retains_current_state_and_prunes_removed(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        spec = EmbeddingSpec(provider="fake", model="m", dimension=8)
        original = CHUNKS[0]
        catalog.mark_embedded(original.chunk_id, spec, _required_hash(original))

        assert catalog.replace_chunks(original.doc_id, [original]) == []
        assert original.chunk_id in catalog.embedding_state(spec.space_id)

        replacement = CatalogChunk(
            chunk_id="INC-1104:new",
            doc_id=original.doc_id,
            text="Replacement payment report procedure",
        )
        removed = catalog.replace_chunks(original.doc_id, [replacement])

        assert removed == [original.chunk_id]
        assert catalog.get_chunks([original.chunk_id]) == []
        assert catalog.embedding_state(spec.space_id) == {}


class TestEmbeddingLifecycleLedger:
    SPEC = EmbeddingSpec(
        provider="openai", model="text-embedding-3-small", dimension=1536
    )
    OTHER_SPEC = EmbeddingSpec(provider="st", model="all-MiniLM-L6-v2", dimension=384)

    def test_all_chunks_stale_before_embedding(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        assert set(catalog.stale_chunk_ids(self.SPEC)) == {
            chunk.chunk_id for chunk in CHUNKS
        }

    def test_mark_embedded_clears_staleness(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        for chunk in CHUNKS:
            catalog.mark_embedded(chunk.chunk_id, self.SPEC, _required_hash(chunk))
        assert catalog.stale_chunk_ids(self.SPEC) == []

    def test_spaces_are_independent(self, catalog: SqliteDocumentCatalog) -> None:
        for chunk in CHUNKS:
            catalog.mark_embedded(chunk.chunk_id, self.SPEC, _required_hash(chunk))
        assert len(catalog.stale_chunk_ids(self.OTHER_SPEC)) == len(CHUNKS)

    def test_content_change_makes_vector_stale(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        catalog.mark_embedded("KB-77:0", self.SPEC, _required_hash(CHUNKS[2]))
        catalog.upsert_chunks(
            [
                CatalogChunk(
                    chunk_id="KB-77:0",
                    doc_id="KB-77",
                    text="Title: rotate credentials\nBody: new procedure",
                )
            ]
        )
        assert "KB-77:0" in catalog.stale_chunk_ids(self.SPEC)

    def test_embedding_state_records_spec_metadata(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        clock = FakeClock()
        with SqliteDocumentCatalog(now=clock) as instance:
            instance.upsert_documents(INCIDENTS[:1])
            instance.upsert_chunks(CHUNKS[:1])
            instance.mark_embedded("INC-1104:0", self.SPEC, _required_hash(CHUNKS[0]))
            states = instance.embedding_state(self.SPEC.space_id)
            state = states["INC-1104:0"]
            assert state.provider == "openai"
            assert state.model == "text-embedding-3-small"
            assert state.dimension == 1536
            assert state.version == "v1"
            assert state.content_hash == CHUNKS[0].content_hash
            assert state.created_at == clock.moment.isoformat()

    def test_embedding_state_filters_by_chunk_ids(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        for chunk in CHUNKS:
            catalog.mark_embedded(chunk.chunk_id, self.SPEC, _required_hash(chunk))
        states = catalog.embedding_state(self.SPEC.space_id, ["KB-77:0"])
        assert set(states) == {"KB-77:0"}
        assert catalog.embedding_state(self.SPEC.space_id, []) == {}

    def test_invalidate_embeddings_removes_every_space_for_chunk(
        self, catalog: SqliteDocumentCatalog
    ) -> None:
        chunk = CHUNKS[0]
        for spec in (self.SPEC, self.OTHER_SPEC):
            catalog.mark_embedded(chunk.chunk_id, spec, _required_hash(chunk))

        catalog.invalidate_embeddings([chunk.chunk_id, chunk.chunk_id])

        assert catalog.embedding_state(self.SPEC.space_id) == {}
        assert catalog.embedding_state(self.OTHER_SPEC.space_id) == {}


class TestDurableBudgetLedger:
    def test_accumulates_within_a_day(self) -> None:
        clock = FakeClock()
        with SqliteDocumentCatalog(now=clock) as catalog:
            catalog.record("openai", 1000, 0.01)
            catalog.record("openai", 500, 0.005)
            assert catalog.spent_today() == Decimal("0.015")
            assert catalog.tokens_today("openai") == 1500

    def test_day_rollover_resets_daily_but_not_monthly(self) -> None:
        clock = FakeClock("2026-08-23T23:00:00+00:00")
        with SqliteDocumentCatalog(now=clock) as catalog:
            catalog.record("openai", 1000, 0.01)
            clock.advance_days(1)
            assert catalog.spent_today() == 0.0
            assert catalog.spent_month() == Decimal("0.01")

    def test_month_rollover_resets_monthly(self) -> None:
        clock = FakeClock("2026-08-31T12:00:00+00:00")
        with SqliteDocumentCatalog(now=clock) as catalog:
            catalog.record("openai", 1000, 0.01)
            clock.advance_days(1)
            assert catalog.spent_month() == 0.0

    def test_rejects_negative_values(self) -> None:
        with SqliteDocumentCatalog() as catalog:
            with pytest.raises(ValueError):
                catalog.record("openai", -1, 0.0)
            with pytest.raises(ValueError):
                catalog.record("openai", 0, -0.1)

    def test_catalog_serves_as_router_ledger(self) -> None:
        with SqliteDocumentCatalog() as catalog:
            router = EmbeddingRouter(
                FakeEmbedding(),
                FakeEmbedding(provider="st", model="fake-local", dimension=8),
                ledger=catalog,
                daily_budget_usd=0.01,
                cost_per_million_tokens=0.02,
            )
            assert router.select(estimated_tokens=0).reason is SelectionReason.PRIMARY
            catalog.record("fake", 600_000, 0.012)
            assert (
                router.select(estimated_tokens=0).reason
                is SelectionReason.BUDGET_DAILY_EXCEEDED
            )

    def test_persists_exact_price_provenance(self) -> None:
        price = EmbeddingPrice.from_usd_per_million(
            "openai",
            "text-embedding-3-small",
            "0.02",
            version="catalog-test-v1",
        )
        with SqliteDocumentCatalog() as catalog:
            catalog.record(price.charge(17))

            record = catalog.usage_records()[0]

            assert record.status is UsageStatus.COMMITTED
            assert record.charge.provider == "openai"
            assert record.charge.model == "text-embedding-3-small"
            assert record.charge.processing_mode == "standard"
            assert record.charge.tokens == 17
            assert record.charge.rate_nanos_per_million == 20_000_000
            assert record.charge.charge_nanos == 340
            assert record.charge.price_version == "catalog-test-v1"
            assert catalog.spent_today() == Decimal("0.00000034")

    def test_sqlite_reservation_is_atomic_across_connections(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "atomic-budget.db"
        with SqliteDocumentCatalog(path):
            pass
        barrier = Barrier(3)

        def select() -> SelectionReason:
            with SqliteDocumentCatalog(path) as ledger:
                router = EmbeddingRouter(
                    FakeEmbedding(),
                    FakeEmbedding(provider="st", model="fallback", dimension=8),
                    ledger=ledger,
                    daily_budget_usd="1.00",
                    cost_per_million_tokens="1.00",
                )
                barrier.wait()
                return router.select(estimated_tokens=600_000).reason

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(select) for _ in range(2)]
            barrier.wait()
            reasons = [future.result() for future in futures]

        assert reasons.count(SelectionReason.PRIMARY) == 1
        assert reasons.count(SelectionReason.BUDGET_DAILY_EXCEEDED) == 1
        with SqliteDocumentCatalog(path) as catalog:
            assert catalog.spent_today() == Decimal("0.6")
            records = catalog.usage_records()
            assert len(records) == 1
            assert records[0].status is UsageStatus.RESERVED

    def test_reconciles_reserved_sqlite_charge(self) -> None:
        with SqliteDocumentCatalog() as catalog:
            router = EmbeddingRouter(
                FakeEmbedding(),
                ledger=catalog,
                daily_budget_usd="1.00",
                cost_per_million_tokens="1.00",
            )
            selection = router.select(estimated_tokens=900_000)
            assert selection.reservation is not None

            router.record_usage(125_000, reservation=selection.reservation)

            assert catalog.spent_today() == Decimal("0.125")
            record = catalog.usage_records()[0]
            assert record.status is UsageStatus.COMMITTED
            assert record.charge.tokens == 125_000


class TestPersistence:
    def test_file_backed_catalog_survives_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.db"
        spec = EmbeddingSpec(provider="openai", model="tes3s", dimension=1536)
        with SqliteDocumentCatalog(path) as catalog:
            catalog.upsert_documents(INCIDENTS)
            catalog.upsert_chunks(CHUNKS)
            catalog.mark_embedded("KB-77:0", spec, _required_hash(CHUNKS[2]))
            catalog.record("openai", 100, 0.001)

        with SqliteDocumentCatalog(path) as reopened:
            assert len(reopened.find()) == 4
            hits = reopened.search_lexical("rotate credentials")
            assert hits[0].chunk_id == "KB-77:0"
            assert "KB-77:0" in reopened.embedding_state(spec.space_id)
            assert reopened.spent_today() == Decimal("0.001")

    def test_legacy_float_ledger_is_migrated(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy-ledger.db"
        connection = sqlite3.connect(path)
        connection.execute(
            """
            CREATE TABLE embedding_usage (
                date TEXT NOT NULL,
                provider TEXT NOT NULL,
                tokens INTEGER NOT NULL,
                estimated_usd REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO embedding_usage VALUES (?, ?, ?, ?)",
            ("2026-08-23", "openai", 500_000, 0.01),
        )
        connection.commit()
        connection.close()

        with SqliteDocumentCatalog(path, now=FakeClock()) as catalog:
            assert catalog.spent_today() == Decimal("0.01")
            record = catalog.usage_records()[0]
            assert record.status is UsageStatus.COMMITTED
            assert record.charge.model == "<legacy>"
            assert record.charge.tokens == 500_000
            assert record.charge.charge_nanos == 10_000_000
            assert record.charge.price_version == "legacy-float-migration"


class TestLexicalUnavailable:
    def test_typed_error_when_fts_missing(self) -> None:
        with SqliteDocumentCatalog() as catalog:
            catalog._fts_available = False
            with pytest.raises(LexicalUnavailableError):
                catalog.search_lexical("anything")
