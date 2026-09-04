from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import pytest

from vectorstore import (
    AzureSqlCatalog,
    AzureSqlDocumentCatalog,
    BudgetLedger,
    BudgetPeriod,
    CatalogChunk,
    CatalogDocument,
    DocumentCatalog,
    EmbeddingPrice,
    EmbeddingSpec,
    LexicalUnavailableError,
    RetrievalScope,
)


class FakeDatabase:
    def __init__(self, *responses: object) -> None:
        self.responses = deque(responses)
        self.executions: list[tuple[str, tuple[object, ...]]] = []
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
        self.autocommit = False

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
        self.database.executions.append((statement, tuple(parameters)))
        response = self.database.responses.popleft() if self.database.responses else []
        if isinstance(response, BaseException):
            raise response
        if not isinstance(response, Iterable):
            raise TypeError("fake database responses must be iterable")
        self.rows = list(response)
        return self

    def fetchone(self) -> Any | None:
        return self.rows.pop(0) if self.rows else None

    def fetchall(self) -> list[Any]:
        rows = self.rows
        self.rows = []
        return rows

    def close(self) -> None:
        self.closed = True


class AzureSqlError(RuntimeError):
    def __init__(self, number: int, message: str) -> None:
        super().__init__(message)
        self.number = number


NOW = datetime(2026, 9, 4, 12, 30, tzinfo=UTC)


def _catalog(database: FakeDatabase) -> AzureSqlDocumentCatalog:
    return AzureSqlDocumentCatalog(connection_factory=database, now=lambda: NOW)


def test_alias_and_runtime_contracts() -> None:
    database = FakeDatabase()
    catalog = AzureSqlCatalog(connection_factory=database)

    assert isinstance(catalog, AzureSqlDocumentCatalog)
    assert isinstance(catalog, DocumentCatalog)
    assert isinstance(catalog, BudgetLedger)
    assert database.connections == []


def test_schema_creation_is_explicit_versioned_idempotent_and_validated() -> None:
    database = FakeDatabase(
        [],
        [(1, "vectorstore_catalog_fulltext", 1033)],
        [],
        [
            (
                "vectorstore_catalog_fulltext",
                "AUTO",
                "PK_vectorstore_catalog_chunks",
                1033,
            )
        ],
    )
    catalog = _catalog(database)

    catalog.create_schema()

    core_ddl, parameters = database.executions[0]
    fulltext_ddl, fulltext_parameters = database.executions[2]
    assert parameters == fulltext_parameters == ()
    ddl = catalog.schema_sql
    assert "IF OBJECT_ID" in ddl
    assert "CREATE TABLE [dbo].[documents]" in ddl
    assert "CREATE TABLE [dbo].[chunks]" in ddl
    assert "CREATE TABLE [dbo].[chunk_embeddings]" in ddl
    assert "CREATE TABLE [dbo].[embedding_usage]" in ddl
    assert "FULLTEXTSERVICEPROPERTY" in ddl
    assert "CREATE FULLTEXT CATALOG [vectorstore_catalog_fulltext]" in ddl
    assert "CREATE FULLTEXT INDEX ON [dbo].[chunks]" in ddl
    assert "CHANGE_TRACKING = AUTO" in ddl
    assert "STOPLIST = OFF" in ddl
    assert "CREATE FULLTEXT" not in core_ddl
    assert "CREATE FULLTEXT CATALOG" in fulltext_ddl
    assert len(database.connections) == 2
    assert database.connections[0].commits == 1
    assert database.connections[0].rollbacks == 0
    assert database.connections[0].closed
    assert database.connections[1].autocommit is True
    assert database.connections[1].commits == 0
    assert database.connections[1].closed


def test_schema_mismatch_rolls_back_create_operation() -> None:
    database = FakeDatabase(
        [],
        [(2, "vectorstore_catalog_fulltext", 1033)],
    )
    catalog = _catalog(database)

    with pytest.raises(RuntimeError, match="schema version 2"):
        catalog.create_schema()

    assert database.connections[0].commits == 0
    assert database.connections[0].rollbacks == 1
    assert database.connections[0].closed


def test_upsert_documents_uses_last_duplicate_and_one_transaction() -> None:
    database = FakeDatabase()
    catalog = _catalog(database)

    catalog.upsert_documents(
        [
            CatalogDocument("doc", title="old"),
            CatalogDocument(
                "doc",
                title="new",
                doc_type="incident",
                tenant_id="acme",
                attributes={"severity": 2, "reviewed": True},
            ),
        ]
    )

    assert len(database.executions) == 1
    statement, parameters = database.executions[0]
    assert "WITH (UPDLOCK, SERIALIZABLE)" in statement
    assert "IF @@ROWCOUNT = 0" in statement
    assert parameters[1:5] == ("new", "incident", "acme", None)
    assert parameters[9] == '{"reviewed":true,"severity":2}'
    assert parameters[10:13] == ("doc", "doc", None)
    assert database.connections[0].commits == 1


def test_chunks_upsert_and_document_replacement_are_transactional() -> None:
    database = FakeDatabase(
        [("old",), ("keep",)],
        [],
        [],
    )
    catalog = _catalog(database)
    chunks = [
        CatalogChunk("keep", "doc", "new text", chunk_index=0),
        CatalogChunk("new", "doc", "another", chunk_index=1),
    ]

    removed = catalog.replace_chunks("doc", chunks)

    assert removed == ["old"]
    select, select_parameters = database.executions[0]
    assert "WITH (UPDLOCK, HOLDLOCK)" in select
    assert select_parameters == ("doc",)
    assert sum("IF @@ROWCOUNT = 0" in sql for sql, _ in database.executions) == 2
    delete, delete_parameters = database.executions[-1]
    assert "DELETE FROM [dbo].[chunks]" in delete
    assert delete_parameters == ("old",)
    assert database.connections[0].commits == 1


def test_find_pushes_scope_first_class_and_json_filters_into_sql() -> None:
    database = FakeDatabase(
        [
            (
                "doc",
                "source.md",
                "Incident",
                "incident",
                "acme",
                "internal",
                "sre",
                "OPEN",
                None,
                None,
                '{"severity":3}',
            )
        ]
    )
    catalog = _catalog(database)

    documents = catalog.find(
        {
            "doc_type": {"$in": ["incident", "runbook"]},
            "severity": {"$gte": 2},
        },
        RetrievalScope(tenant_id="acme", visibility=("internal", "public")),
        limit=5,
    )

    statement, parameters = database.executions[0]
    assert "SELECT TOP (?)" in statement
    assert "d.[tenant_id] IS NULL OR d.[tenant_id] = ?" in statement
    assert "d.[visibility] IN (?, ?)" in statement
    assert "d.[doc_type] IN (?, ?)" in statement
    assert "OPENJSON(d.[attributes_json])" in statement
    assert "TRY_CONVERT(FLOAT, attribute_item.[value]) >= ?" in statement
    assert parameters == (
        5,
        "acme",
        "internal",
        "public",
        "incident",
        "runbook",
        "severity",
        2,
    )
    assert documents == [
        CatalogDocument(
            "doc",
            source="source.md",
            title="Incident",
            doc_type="incident",
            tenant_id="acme",
            visibility="internal",
            owner_group="sre",
            status="OPEN",
            attributes={"severity": 3},
        )
    ]


def test_lexical_search_uses_parameterized_containstable_and_stable_ranks() -> None:
    database = FakeDatabase([("b", 900), ("a", 500)])
    catalog = _catalog(database)

    hits = catalog.search_lexical(
        'INC-1104 "payment report"',
        k=2,
        filter={"status": "OPEN"},
        scope=RetrievalScope(tenant_id="acme"),
    )

    statement, parameters = database.executions[0]
    assert "FROM CONTAINSTABLE(" in statement
    assert "LANGUAGE 1033" in statement
    assert "JOIN [dbo].[documents] AS d" in statement
    assert "c.[active] = 1" in statement
    assert "ORDER BY fulltext_result.[RANK] DESC, c.[chunk_id] ASC" in statement
    assert parameters == (
        2,
        '"payment report" AND "INC-1104"',
        "acme",
        "OPEN",
    )
    assert [(hit.chunk_id, hit.rank, hit.score) for hit in hits] == [
        ("b", 1, 900.0),
        ("a", 2, 500.0),
    ]


def test_missing_fulltext_index_has_typed_degradation_error() -> None:
    database = FakeDatabase(
        AzureSqlError(
            7601,
            "Cannot use CONTAINS because table is not full-text indexed",
        )
    )
    catalog = _catalog(database)

    with pytest.raises(LexicalUnavailableError, match="Azure SQL lexical"):
        catalog.search_lexical("payment")

    assert database.connections[0].closed


def test_get_chunks_preserves_order_and_parses_bit_values() -> None:
    database = FakeDatabase(
        [
            ("a", "doc", 0, None, "Alpha", "hash-a", 1),
            ("b", "doc", 1, "Details", "Beta", "hash-b", "false"),
        ]
    )
    catalog = _catalog(database)

    chunks = catalog.get_chunks(["b", "missing", "a", "b"])

    assert [chunk.chunk_id for chunk in chunks] == ["b", "a", "b"]
    assert chunks[0].active is False
    assert chunks[1].active is True


def test_embedding_lifecycle_queries_and_upsert() -> None:
    spec = EmbeddingSpec("openai", "model", 3)
    database = FakeDatabase(
        [
            (
                "chunk",
                spec.space_id,
                "openai",
                "model",
                3,
                "v1",
                "abc",
                NOW,
            )
        ],
        [],
        [("stale",)],
    )
    catalog = _catalog(database)

    state = catalog.embedding_state(spec.space_id, ["chunk"])
    catalog.mark_embedded("chunk", spec, "abc")
    stale = catalog.stale_chunk_ids(spec)

    assert state["chunk"].created_at == NOW.isoformat()
    mark_statement, mark_parameters = database.executions[1]
    assert "WITH (UPDLOCK, SERIALIZABLE)" in mark_statement
    assert mark_parameters[6:8] == ("chunk", spec.space_id)
    assert stale == ["stale"]
    assert len(database.connections) == 3
    assert database.connections[1].commits == 1


def test_budget_reservation_uses_serialized_schema_lock() -> None:
    price = EmbeddingPrice.from_usd_per_million(
        "openai", "model", "0.02", version="pricing-v1"
    )
    charge = price.charge(100)
    database = FakeDatabase(
        [(1,)],
        [],
        [(0,)],
        [(0,)],
        [],
    )
    catalog = _catalog(database)

    decision = catalog.reserve(
        charge,
        daily_limit_nanos=10_000,
        monthly_limit_nanos=10_000,
        ttl_seconds=30,
    )

    assert decision.reservation is not None
    lock_statement, _ = database.executions[0]
    assert "WITH (UPDLOCK, HOLDLOCK)" in lock_statement
    insert_statement, insert_parameters = database.executions[-1]
    assert "INSERT INTO [dbo].[embedding_usage]" in insert_statement
    assert insert_parameters[1:6] == (
        "2026-09-04",
        "openai",
        "model",
        "standard",
        100,
    )
    assert database.connections[0].commits == 1


def test_budget_rejection_commits_expiry_work_without_inserting() -> None:
    charge = EmbeddingPrice.from_usd_per_million(
        "openai", "model", "1.00", version="pricing-v1"
    ).charge(200_000)
    database = FakeDatabase([(1,)], [], [(900_000_000,)])
    catalog = _catalog(database)

    decision = catalog.reserve(
        charge,
        daily_limit_nanos=1_000_000_000,
        monthly_limit_nanos=None,
        ttl_seconds=30,
    )

    assert decision.exceeded is BudgetPeriod.DAILY
    assert decision.reservation is None
    assert len(database.executions) == 3
    assert database.connections[0].commits == 1


def test_write_failure_rolls_back_and_closes_connection() -> None:
    database = FakeDatabase(RuntimeError("database unavailable"))
    catalog = _catalog(database)

    with pytest.raises(RuntimeError, match="database unavailable"):
        catalog.upsert_documents([CatalogDocument("doc")])

    assert database.connections[0].commits == 0
    assert database.connections[0].rollbacks == 1
    assert database.connections[0].closed


def test_configuration_validation_precedes_connection() -> None:
    database = FakeDatabase()

    with pytest.raises(ValueError, match="schema_name"):
        AzureSqlDocumentCatalog(
            connection_factory=database,
            schema_name="dbo]; DROP TABLE x;--",
        )
    with pytest.raises(ValueError, match="fulltext_catalog_name"):
        AzureSqlDocumentCatalog(
            connection_factory=database,
            fulltext_catalog_name="bad name",
        )
    with pytest.raises(ValueError, match="language_lcid"):
        AzureSqlDocumentCatalog(connection_factory=database, language_lcid=0)
    with pytest.raises(ValueError, match="either connection_string"):
        AzureSqlDocumentCatalog(
            "Server=example",
            connection_factory=database,
        )

    assert database.connections == []


def test_empty_operations_do_not_connect() -> None:
    database = FakeDatabase()
    catalog = _catalog(database)

    catalog.upsert_documents([])
    catalog.upsert_chunks([])
    catalog.delete_documents([])
    catalog.invalidate_embeddings([])
    assert catalog.get_chunks([]) == []
    assert catalog.embedding_state("space", []) == {}

    assert database.connections == []
