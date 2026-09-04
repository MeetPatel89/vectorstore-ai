from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import pytest

from vectorstore import (
    CatalogChunk,
    CatalogDocument,
    DocumentCatalog,
    PostgresDocumentCatalog,
    RetrievalScope,
)
from vectorstore.catalog import LexicalUnavailableError
from vectorstore.embeddings import (
    BudgetLedger,
    BudgetPeriod,
    EmbeddingPrice,
)


class FakeDatabase:
    def __init__(self, *responses: object) -> None:
        self.responses = deque(responses)
        self.executions: list[
            tuple[str, str, tuple[object, ...] | list[tuple[object, ...]]]
        ] = []
        self.connections: list[FakeConnection] = []

    def __call__(self) -> FakeConnection:
        connection = FakeConnection(self)
        self.connections.append(connection)
        return connection


class FakeConnection:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.database)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


class FakeCursor:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database
        self.rows: list[Any] = []
        self.closed = False

    def execute(
        self, statement: str, parameters: tuple[object, ...] = ()
    ) -> FakeCursor:
        self.database.executions.append(("execute", statement, tuple(parameters)))
        self._set_response()
        return self

    def executemany(
        self, statement: str, parameters: Iterable[tuple[object, ...]]
    ) -> FakeCursor:
        rows = list(parameters)
        self.database.executions.append(("executemany", statement, rows))
        self._set_response()
        return self

    def fetchone(self) -> Any | None:
        return self.rows.pop(0) if self.rows else None

    def fetchall(self) -> list[Any]:
        rows = self.rows
        self.rows = []
        return rows

    def close(self) -> None:
        self.closed = True

    def _set_response(self) -> None:
        response = self.database.responses.popleft() if self.database.responses else []
        if isinstance(response, BaseException):
            raise response
        if not isinstance(response, Iterable):
            raise TypeError("fake database responses must be iterable")
        self.rows = list(response)


class FakePostgresError(RuntimeError):
    def __init__(self, message: str, sqlstate: str) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


def _catalog(
    database: FakeDatabase,
    *,
    now: datetime | None = None,
) -> PostgresDocumentCatalog:
    clock = (lambda: now) if now is not None else None
    if clock is None:
        return PostgresDocumentCatalog(connection_factory=database)
    return PostgresDocumentCatalog(connection_factory=database, now=clock)


def test_catalog_satisfies_document_and_budget_protocols_without_connecting() -> None:
    database = FakeDatabase()
    catalog = _catalog(database)

    assert isinstance(catalog, DocumentCatalog)
    assert isinstance(catalog, BudgetLedger)
    assert database.connections == []


def test_schema_configuration_is_read_only_after_validation() -> None:
    catalog = PostgresDocumentCatalog(
        schema_name="tenant_a",
        text_search_config="pg_catalog.simple",
        connection_factory=FakeDatabase(),
    )

    assert catalog.schema_name == "tenant_a"
    assert catalog.text_search_config == "pg_catalog.simple"
    with pytest.raises(AttributeError):
        setattr(catalog, "schema_name", "tenant_b")
    with pytest.raises(AttributeError):
        setattr(catalog, "text_search_config", "english")


def test_schema_uses_generated_tsvector_and_gin_index() -> None:
    database = FakeDatabase(
        [],
        [(1, "pg_catalog.simple")],
        [("s", "tsvector")],
        [
            (
                "gin",
                "CREATE INDEX idx_chunks_search_vector "
                "ON tenant_a.chunks USING gin (search_vector)",
            )
        ],
    )
    catalog = PostgresDocumentCatalog(
        schema_name="tenant_a",
        text_search_config="pg_catalog.simple",
        connection_factory=database,
    )

    catalog.create_schema()

    ddl = database.executions[0][1]
    assert 'CREATE SCHEMA IF NOT EXISTS "tenant_a"' in ddl
    assert "search_vector TSVECTOR GENERATED ALWAYS AS" in ddl
    assert "to_tsvector('pg_catalog.simple'::regconfig, text)" in ddl
    assert "USING GIN (search_vector)" in ddl
    assert "attributes_json JSONB" in ddl
    assert "ON DELETE CASCADE" in ddl
    assert database.connections[0].commits == 1
    assert database.connections[0].rollbacks == 0
    assert database.connections[0].closed


def test_schema_validation_rejects_missing_gin_index_and_rolls_back() -> None:
    database = FakeDatabase([], [(1, "simple")], [("s", "tsvector")], [])
    catalog = _catalog(database)

    with pytest.raises(RuntimeError, match="missing its GIN lexical index"):
        catalog.create_schema()

    assert database.connections[0].commits == 0
    assert database.connections[0].rollbacks == 1


def test_schema_validation_rejects_text_search_configuration_mismatch() -> None:
    database = FakeDatabase([], [(1, "english")])
    catalog = _catalog(database)

    with pytest.raises(ValueError, match="configuration mismatch"):
        catalog.create_schema()

    assert database.connections[0].commits == 0
    assert database.connections[0].rollbacks == 1


def test_document_and_chunk_upserts_are_transactional() -> None:
    database = FakeDatabase([], [])
    catalog = _catalog(database)
    document = CatalogDocument(
        doc_id="INC-1104",
        title="Missing reports",
        tenant_id="acme",
        attributes={"severity": 3, "customer_visible": False},
    )
    chunk = CatalogChunk(
        chunk_id="INC-1104:0",
        doc_id="INC-1104",
        text="SQLSTATE 23505 while producing payment reports",
    )

    catalog.upsert_documents([document])
    catalog.upsert_chunks([chunk])

    document_execution = database.executions[0]
    assert document_execution[0] == "executemany"
    assert "ON CONFLICT (doc_id) DO UPDATE" in document_execution[1]
    document_rows = document_execution[2]
    assert isinstance(document_rows, list)
    assert document_rows[0][-1] == '{"customer_visible":false,"severity":3}'
    chunk_execution = database.executions[1]
    assert chunk_execution[0] == "executemany"
    assert "search_vector" not in chunk_execution[1]
    assert "ON CONFLICT (chunk_id) DO UPDATE" in chunk_execution[1]
    assert all(connection.commits == 1 for connection in database.connections)


def test_find_pushes_scope_and_typed_json_filters_into_postgresql() -> None:
    database = FakeDatabase(
        [
            (
                "INC-1104",
                "incidents",
                "Missing reports",
                "incident",
                "acme",
                "internal",
                "payments",
                "OPEN",
                None,
                None,
                {"severity": 3, "service": "reporting"},
            )
        ]
    )
    catalog = _catalog(database)

    documents = catalog.find(
        {
            "status": "OPEN",
            "service": {"$in": ["reporting", "billing"]},
            "severity": {"$gte": 2},
        },
        scope=RetrievalScope(
            tenant_id="acme",
            visibility=("internal", "public"),
        ),
        limit=5,
    )

    statement = database.executions[0][1]
    parameters = database.executions[0][2]
    assert "d.tenant_id IS NULL OR d.tenant_id = %s" in statement
    assert "d.visibility IS NULL OR d.visibility IN (%s, %s)" in statement
    assert "d.attributes_json -> %s IN (%s::jsonb, %s::jsonb)" in statement
    assert "jsonb_typeof(d.attributes_json -> %s) = 'number'" in statement
    assert parameters == (
        "acme",
        "internal",
        "public",
        "OPEN",
        "service",
        '"reporting"',
        '"billing"',
        "severity",
        "severity",
        2,
        5,
    )
    assert documents == [
        CatalogDocument(
            doc_id="INC-1104",
            source="incidents",
            title="Missing reports",
            doc_type="incident",
            tenant_id="acme",
            visibility="internal",
            owner_group="payments",
            status="OPEN",
            attributes={"severity": 3, "service": "reporting"},
        )
    ]


def test_lexical_search_uses_inverted_index_safe_query_and_rank() -> None:
    database = FakeDatabase(
        [
            ("INC-1104:0", 0.8),
            ("INC-2001:0", 0.25),
        ]
    )
    catalog = _catalog(database)
    raw_query = '"payment reports" OR SQLSTATE 23505); DROP TABLE chunks;--'

    hits = catalog.search_lexical(
        raw_query,
        k=2,
        filter={"status": "OPEN"},
        scope=RetrievalScope(tenant_id="acme"),
    )

    statement = database.executions[0][1]
    parameters = database.executions[0][2]
    assert "websearch_to_tsquery(" in statement
    assert "'simple'::regconfig, %s" in statement
    assert "c.search_vector @@ search_query.value" in statement
    assert "ts_rank_cd(c.search_vector, search_query.value, 32)" in statement
    assert "c.active = TRUE" in statement
    assert "ORDER BY score DESC, c.chunk_id ASC" in statement
    assert raw_query not in statement
    assert parameters == (raw_query, "acme", "OPEN", 2)
    assert [hit.chunk_id for hit in hits] == ["INC-1104:0", "INC-2001:0"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert [hit.score for hit in hits] == [0.8, 0.25]


def test_lexical_schema_error_is_typed_for_retriever_degradation() -> None:
    database = FakeDatabase(
        FakePostgresError('relation "public.chunks" does not exist', "42P01")
    )
    catalog = _catalog(database)

    with pytest.raises(LexicalUnavailableError, match="lexical search is unavailable"):
        catalog.search_lexical("INC-1104")

    assert database.connections[0].closed


def test_get_chunks_preserves_order_and_embedding_state_maps_native_rows() -> None:
    created_at = datetime(2026, 9, 3, 12, tzinfo=UTC)
    database = FakeDatabase(
        [
            (
                "a",
                "doc-a",
                0,
                None,
                "Alpha",
                "hash-a",
                True,
            ),
            (
                "b",
                "doc-b",
                1,
                "body",
                "Beta",
                "hash-b",
                False,
            ),
        ],
        [
            (
                "a",
                "fake__m__8__v1",
                "fake",
                "m",
                8,
                "v1",
                "hash-a",
                created_at,
            )
        ],
    )
    catalog = _catalog(database)

    chunks = catalog.get_chunks(["b", "missing", "a", "b"])
    states = catalog.embedding_state("fake__m__8__v1", ["a"])

    assert [chunk.chunk_id for chunk in chunks] == ["b", "a", "b"]
    assert not chunks[0].active
    assert states["a"].created_at == "2026-09-03T12:00:00+00:00"
    assert states["a"].dimension == 8


def test_reserve_serializes_on_schema_row_and_inserts_auditable_hold() -> None:
    moment = datetime(2026, 9, 3, 12, tzinfo=UTC)
    price = EmbeddingPrice.from_usd_per_million(
        "openai",
        "text-embedding-3-small",
        "1.00",
        version="test-v1",
    )
    charge = price.charge(600_000)
    database = FakeDatabase(
        [(1,)],
        [],
        [(100_000_000,)],
        [(200_000_000,)],
        [],
    )
    catalog = _catalog(database, now=moment)

    decision = catalog.reserve(
        charge,
        daily_limit_nanos=1_000_000_000,
        monthly_limit_nanos=1_000_000_000,
        ttl_seconds=300,
    )

    assert decision.exceeded is None
    assert decision.reservation is not None
    assert decision.reservation.charge == charge
    statements = [execution[1] for execution in database.executions]
    assert "WHERE singleton = TRUE FOR UPDATE" in statements[0]
    assert "status IN (%s, %s)" in statements[2]
    assert "status IN (%s, %s)" in statements[3]
    assert statements[4].startswith('INSERT INTO "public".embedding_usage')
    assert database.connections[0].commits == 1


def test_reserve_rejects_over_daily_limit_without_inserting() -> None:
    moment = datetime(2026, 9, 3, 12, tzinfo=UTC)
    charge = EmbeddingPrice.from_usd_per_million(
        "openai",
        "model",
        "1.00",
        version="test-v1",
    ).charge(200_000)
    database = FakeDatabase([(1,)], [], [(900_000_000,)])
    catalog = _catalog(database, now=moment)

    decision = catalog.reserve(
        charge,
        daily_limit_nanos=1_000_000_000,
        monthly_limit_nanos=None,
        ttl_seconds=300,
    )

    assert decision.exceeded is BudgetPeriod.DAILY
    assert decision.reservation is None
    assert len(database.executions) == 3
    assert database.connections[0].commits == 1


def test_write_failure_rolls_back_and_empty_operations_do_not_connect() -> None:
    database = FakeDatabase(RuntimeError("database unavailable"))
    catalog = _catalog(database)

    catalog.upsert_documents([])
    catalog.upsert_chunks([])
    catalog.delete_documents([])
    assert catalog.get_chunks([]) == []
    assert catalog.embedding_state("space", []) == {}
    assert database.connections == []

    with pytest.raises(RuntimeError, match="database unavailable"):
        catalog.upsert_documents([CatalogDocument(doc_id="doc")])

    assert database.connections[0].commits == 0
    assert database.connections[0].rollbacks == 1
    assert database.connections[0].closed


def test_rejects_unsafe_schema_configuration_and_ambiguous_connection() -> None:
    database = FakeDatabase()

    with pytest.raises(ValueError, match="schema_name"):
        PostgresDocumentCatalog(
            schema_name='public"; DROP SCHEMA public;--',
            connection_factory=database,
        )
    with pytest.raises(ValueError, match="text_search_config"):
        PostgresDocumentCatalog(
            text_search_config="simple); DROP TABLE chunks;--",
            connection_factory=database,
        )
    with pytest.raises(ValueError, match="either connection_string"):
        PostgresDocumentCatalog(
            "postgresql://example/test",
            connection_factory=database,
        )

    assert database.connections == []
