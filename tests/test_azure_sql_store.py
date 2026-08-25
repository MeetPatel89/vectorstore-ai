from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

import pytest

from vectorstore import AzureSqlVectorStore, Chunk, create_store


class FakeDatabase:
    def __init__(self, *responses: object) -> None:
        self.responses = deque(responses)
        self.executions: list[tuple[str, tuple[object, ...]]] = []
        self.connections: list[FakeConnection] = []

    def __call__(self) -> "FakeConnection":
        connection = FakeConnection(self)
        self.connections.append(connection)
        return connection


class FakeConnection:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> "FakeCursor":
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
    ) -> "FakeCursor":
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


def _store(database: FakeDatabase, dimension: int = 3) -> AzureSqlVectorStore:
    return AzureSqlVectorStore(dimension, connection_factory=database)


def test_schema_creation_is_explicit_idempotent_and_validated() -> None:
    database = FakeDatabase([], [(3, 0)])
    store = _store(database)

    store.create_schema()

    ddl, ddl_parameters = database.executions[0]
    validation, validation_parameters = database.executions[1]
    assert "IF OBJECT_ID" in ddl
    assert "CREATE TABLE [dbo].[vectorstore_chunks]" in ddl
    assert "[embedding] VECTOR(3) NOT NULL" in ddl
    assert "Latin1_General_100_BIN2" in ddl
    assert ddl_parameters == ()
    assert "sys.columns" in validation
    assert validation_parameters == ("[dbo].[vectorstore_chunks]",)
    assert database.connections[0].commits == 1
    assert database.connections[0].rollbacks == 0
    assert database.connections[0].closed


def test_schema_validation_rejects_wrong_dimension_and_rolls_back() -> None:
    database = FakeDatabase([], [(4, 0)])
    store = _store(database)

    with pytest.raises(ValueError, match=r"expected float32 VECTOR\(3\)"):
        store.create_schema()

    assert database.connections[0].commits == 0
    assert database.connections[0].rollbacks == 1


def test_upsert_uses_last_duplicate_and_commits_one_transaction() -> None:
    database = FakeDatabase()
    store = _store(database)

    store.upsert(
        [
            Chunk("same", "First", {"version": 1}),
            Chunk("same", "Last", {"version": 2, "active": True}),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    assert len(database.executions) == 1
    statement, parameters = database.executions[0]
    assert "WITH (UPDLOCK, SERIALIZABLE)" in statement
    assert "IF @@ROWCOUNT = 0" in statement
    assert parameters == (
        "Last",
        '{"version":2,"active":true}',
        "[0.0,1.0,0.0]",
        "same",
        "same",
        "Last",
        '{"version":2,"active":true}',
        "[0.0,1.0,0.0]",
    )
    assert database.connections[0].commits == 1


def test_write_error_rolls_back_and_closes_connection() -> None:
    database = FakeDatabase(RuntimeError("database unavailable"))
    store = _store(database)

    with pytest.raises(RuntimeError, match="database unavailable"):
        store.upsert([Chunk("one", "One")], [[1.0, 0.0, 0.0]])

    assert database.connections[0].commits == 0
    assert database.connections[0].rollbacks == 1
    assert database.connections[0].closed


def test_search_uses_exact_cosine_and_translates_filters_before_top_k() -> None:
    database = FakeDatabase(
        [
            ("one", "One", '{"kind":"note","rank":3}', 1.0000001),
            ("two", "Two", '{"kind":"guide","rank":2}', 0.25),
        ]
    )
    store = _store(database)

    results = store.search(
        [1.0, 0.0, 0.0],
        k=2,
        filter={
            "kind": {"$in": ["note", "guide"]},
            "rank": {"$gte": 2},
            "active": True,
        },
    )

    statement, parameters = database.executions[0]
    assert "VECTOR_DISTANCE('cosine'" in statement
    assert "OPENJSON(records.[metadata_json])" in statement
    assert "COLLATE Latin1_General_100_BIN2 = ?" in statement
    assert "TRY_CONVERT(FLOAT, metadata_item.[value]) >= ?" in statement
    assert "ORDER BY [score] DESC, records.[chunk_id] ASC" in statement
    assert parameters == (
        2,
        "[1.0,0.0,0.0]",
        "kind",
        "note",
        "guide",
        "rank",
        2,
        "active",
        "true",
    )
    assert [result.chunk.id for result in results] == ["one", "two"]
    assert [result.score for result in results] == [1.0, 0.25]
    assert database.connections[0].commits == 0


def test_zero_query_returns_deterministic_zero_scores_without_distance_call() -> None:
    database = FakeDatabase([("a", "A", "{}", 0.0)])
    store = _store(database)

    results = store.search([0.0, 0.0, 0.0], k=1)

    statement, parameters = database.executions[0]
    assert "VECTOR_DISTANCE" not in statement
    assert "CAST(0.0 AS FLOAT)" in statement
    assert parameters == (1,)
    assert results[0].score == 0.0


def test_get_preserves_order_and_count_uses_separate_connections() -> None:
    database = FakeDatabase(
        [
            ("alpha", "Alpha", '{"rank":1}'),
            ("gamma", "Gamma", '{"active":true}'),
        ],
        [(2,)],
    )
    store = _store(database)

    chunks = store.get(["gamma", "missing", "alpha", "gamma"])
    count = store.count()

    assert chunks == [
        Chunk("gamma", "Gamma", {"active": True}),
        Chunk("alpha", "Alpha", {"rank": 1}),
        Chunk("gamma", "Gamma", {"active": True}),
    ]
    assert count == 2
    assert len(database.connections) == 2
    assert all(connection.closed for connection in database.connections)


def test_delete_deduplicates_ids() -> None:
    database = FakeDatabase()
    store = _store(database)

    store.delete(["one", "two", "one"])

    statement, parameters = database.executions[0]
    assert "DELETE FROM [dbo].[vectorstore_chunks]" in statement
    assert parameters == ("one", "two")
    assert database.connections[0].commits == 1


@pytest.mark.parametrize("dimension", [0, 1999, True])
def test_rejects_dimensions_unsupported_by_azure_sql(dimension: int) -> None:
    database = FakeDatabase()

    with pytest.raises(ValueError, match="between 1 and 1998"):
        AzureSqlVectorStore(dimension, connection_factory=database)

    assert database.connections == []


def test_rejects_unsafe_identifiers_and_ambiguous_connections() -> None:
    database = FakeDatabase()

    with pytest.raises(ValueError, match="schema_name"):
        AzureSqlVectorStore(
            3,
            schema_name="dbo]; DROP TABLE x;--",
            connection_factory=database,
        )
    with pytest.raises(ValueError, match="either connection_string"):
        AzureSqlVectorStore(
            3,
            connection_string="Server=example",
            connection_factory=database,
        )


def test_validates_vectors_and_filters_before_connecting() -> None:
    database = FakeDatabase()
    store = _store(database)

    with pytest.raises(ValueError, match="store dimension"):
        store.upsert([Chunk("bad", "Bad")], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="finite numeric"):
        store.search([1.0, 0.0, 0.0], filter={"rank": {"$gte": "high"}})
    with pytest.raises(ValueError, match="UTF-16"):
        store.upsert([Chunk("😀" * 226, "Too long")], [[1.0, 0.0, 0.0]])

    assert database.connections == []


def test_registered_factory_does_not_require_driver_when_factory_is_injected() -> None:
    database = FakeDatabase()

    store = create_store(
        "azure-sql",
        dimension=3,
        connection_factory=database,
    )

    assert isinstance(store, AzureSqlVectorStore)


def test_empty_operations_do_not_open_connections() -> None:
    database = FakeDatabase()
    store = _store(database)

    store.upsert([], [])
    store.delete([])
    assert store.search([1.0, 0.0, 0.0], k=0) == []
    assert store.get([]) == []

    assert database.connections == []
