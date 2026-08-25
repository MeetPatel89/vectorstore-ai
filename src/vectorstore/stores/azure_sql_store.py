"""Azure SQL vector store backed by the native ``VECTOR`` data type."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, override

import numpy as np

from vectorstore.models import Chunk, MetadataFilter, MetadataValue, SearchResult

from .base import VectorStore

ConnectionFactory = Callable[[], Any]

_AZURE_SQL_VECTOR_MAX_DIMENSION = 1998
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ORDERED_OPERATORS = {
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


class AzureSqlVectorStore(VectorStore):
    """An exact cosine-similarity store using Azure SQL native vectors.

    A new DB-API connection is opened for each public operation, making store
    instances safe to share between request handlers without sharing cursors or
    transactions. By default, the connection string is read from
    ``AZURE_SQL_CONNECTIONSTRING`` and schema DDL is not run automatically.

    ``connection_factory`` is primarily useful for dependency injection and
    tests. Production callers normally use Microsoft's optional
    ``mssql-python`` driver through a connection string.
    """

    def __init__(
        self,
        dimension: int,
        connection_string: str | None = None,
        *,
        schema_name: str = "dbo",
        table_name: str = "vectorstore_chunks",
        initialize_schema: bool = False,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or not 1 <= dimension <= _AZURE_SQL_VECTOR_MAX_DIMENSION
        ):
            raise ValueError(
                "dimension must be between 1 and "
                f"{_AZURE_SQL_VECTOR_MAX_DIMENSION} for Azure SQL VECTOR"
            )
        self._validate_identifier(schema_name, "schema_name")
        self._validate_identifier(table_name, "table_name")

        if connection_factory is not None and connection_string is not None:
            raise ValueError(
                "pass either connection_string or connection_factory, not both"
            )
        if connection_factory is None:
            resolved_connection_string = connection_string or os.environ.get(
                "AZURE_SQL_CONNECTIONSTRING"
            )
            if not resolved_connection_string:
                raise ValueError(
                    "Azure SQL connection string is required; pass connection_string "
                    "or set AZURE_SQL_CONNECTIONSTRING"
                )
            connection_factory = _mssql_connection_factory(resolved_connection_string)
        elif not callable(connection_factory):
            raise TypeError("connection_factory must be callable")

        self._dimension = dimension
        self.schema_name = schema_name
        self.table_name = table_name
        self._qualified_table = (
            f"{_quote_identifier(schema_name)}.{_quote_identifier(table_name)}"
        )
        self._connection_factory = connection_factory

        if initialize_schema:
            self.create_schema()

    @property
    @override
    def dimension(self) -> int:
        """The vector width required by the Azure SQL table."""
        return self._dimension

    @property
    def schema_sql(self) -> str:
        """Idempotent DDL for the store table.

        Run this with a deployment identity. Runtime identities only need
        SELECT, INSERT, UPDATE, and DELETE permissions on the resulting table.
        """
        return f"""
IF OBJECT_ID(N'{self._qualified_table}', N'U') IS NULL
BEGIN
    CREATE TABLE {self._qualified_table}
    (
        [chunk_id] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NOT NULL,
        [chunk_text] NVARCHAR(MAX) NOT NULL,
        [metadata_json] NVARCHAR(MAX) NOT NULL
            CHECK (ISJSON([metadata_json]) = 1),
        [embedding] VECTOR({self._dimension}) NOT NULL,
        CONSTRAINT {_quote_identifier(_primary_key_name(self.table_name))}
            PRIMARY KEY CLUSTERED ([chunk_id])
    );
END;
""".strip()

    def create_schema(self) -> None:
        """Create the backing table if needed and validate its vector shape."""
        with self._cursor(write=True) as cursor:
            cursor.execute(self.schema_sql)
            self._validate_schema_cursor(cursor)

    def validate_schema(self) -> None:
        """Verify that the table exists with the configured float32 dimension."""
        with self._cursor() as cursor:
            self._validate_schema_cursor(cursor)

    @override
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert new chunks and replace existing chunks with matching IDs."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        prepared_vectors = [
            self._serialize_vector(vector, "vector")[0] for vector in vectors
        ]
        positions: dict[str, int] = {}
        unique_records: list[tuple[Chunk, str, str]] = []
        for chunk, vector_json in zip(chunks, prepared_vectors, strict=True):
            self._validate_chunk(chunk)
            metadata_json = _serialize_metadata(chunk.metadata)
            record = (chunk, metadata_json, vector_json)
            position = positions.get(chunk.id)
            if position is None:
                positions[chunk.id] = len(unique_records)
                unique_records.append(record)
            else:
                unique_records[position] = record

        statement = f"""
UPDATE {self._qualified_table} WITH (UPDLOCK, SERIALIZABLE)
SET [chunk_text] = ?,
    [metadata_json] = ?,
    [embedding] = CAST(? AS VECTOR({self._dimension}))
WHERE [chunk_id] = ?;
IF @@ROWCOUNT = 0
BEGIN
    INSERT INTO {self._qualified_table}
        ([chunk_id], [chunk_text], [metadata_json], [embedding])
    VALUES (?, ?, ?, CAST(? AS VECTOR({self._dimension})));
END;
""".strip()

        with self._cursor(write=True) as cursor:
            for chunk, metadata_json, vector_json in unique_records:
                cursor.execute(
                    statement,
                    (
                        chunk.text,
                        metadata_json,
                        vector_json,
                        chunk.id,
                        chunk.id,
                        chunk.text,
                        metadata_json,
                        vector_json,
                    ),
                )

    @override
    def delete(self, ids: list[str]) -> None:
        """Delete chunks with the requested IDs when present."""
        unique_ids = list(dict.fromkeys(ids))
        if not unique_ids:
            return

        with self._cursor(write=True) as cursor:
            for batch in _batches(unique_ids):
                placeholders = ", ".join("?" for _ in batch)
                cursor.execute(
                    f"DELETE FROM {self._qualified_table} "
                    f"WHERE [chunk_id] IN ({placeholders});",
                    tuple(batch),
                )

    @override
    def search(
        self,
        vector: list[float],
        k: int = 5,
        filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks matching the optional filter."""
        if k <= 0:
            return []

        vector_json, is_zero = self._serialize_vector(vector, "query vector")
        where_sql, filter_parameters = _translate_filter(filter)
        where_clause = f"\nWHERE {where_sql}" if where_sql else ""

        parameters: list[Any]
        if is_zero:
            score_expression = "CAST(0.0 AS FLOAT)"
            parameters = [k]
        else:
            score_expression = (
                "CAST(1.0 - COALESCE(VECTOR_DISTANCE('cosine', "
                f"CAST(? AS VECTOR({self._dimension})), records.[embedding]), "
                "1.0) AS FLOAT)"
            )
            parameters = [k, vector_json]
        parameters.extend(filter_parameters)

        statement = f"""
SELECT TOP (?)
    records.[chunk_id],
    records.[chunk_text],
    records.[metadata_json],
    {score_expression} AS [score]
FROM {self._qualified_table} AS records{where_clause}
ORDER BY [score] DESC, records.[chunk_id] ASC;
""".strip()

        with self._cursor() as cursor:
            cursor.execute(statement, tuple(parameters))
            rows = cursor.fetchall()
        return [_search_result_from_row(row) for row in rows]

    @override
    def get(self, ids: list[str]) -> list[Chunk]:
        """Return known chunks in requested-ID order."""
        if not ids:
            return []

        unique_ids = list(dict.fromkeys(ids))
        by_id: dict[str, Chunk] = {}
        with self._cursor() as cursor:
            for batch in _batches(unique_ids):
                placeholders = ", ".join("?" for _ in batch)
                cursor.execute(
                    f"SELECT [chunk_id], [chunk_text], [metadata_json] "
                    f"FROM {self._qualified_table} "
                    f"WHERE [chunk_id] IN ({placeholders});",
                    tuple(batch),
                )
                for row in cursor.fetchall():
                    chunk = _chunk_from_row(row)
                    by_id[chunk.id] = chunk
        return [by_id[id_] for id_ in ids if id_ in by_id]

    @override
    def count(self) -> int:
        """Return the number of stored chunks."""
        with self._cursor() as cursor:
            cursor.execute(f"SELECT COUNT_BIG(*) FROM {self._qualified_table};")
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Azure SQL count query returned no row")
        return int(_row_value(row, 0, "count"))

    def _serialize_vector(
        self,
        vector: list[float],
        label: str,
    ) -> tuple[str, bool]:
        try:
            values = np.asarray(vector, dtype=np.float32)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} must be a one-dimensional numeric vector"
            ) from exc
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{label} must be a non-empty one-dimensional vector")
        if values.size != self._dimension:
            raise ValueError(
                f"{label} dimension {values.size} does not match store dimension "
                f"{self._dimension}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} must contain only finite values")

        serialized = json.dumps(
            [float(value) for value in values],
            separators=(",", ":"),
            allow_nan=False,
        )
        return serialized, not bool(np.any(values))

    def _validate_schema_cursor(self, cursor: Any) -> None:
        cursor.execute(
            "SELECT [vector_dimensions], [vector_base_type] "
            "FROM sys.columns "
            "WHERE [object_id] = OBJECT_ID(?) AND [name] = N'embedding';",
            (self._qualified_table,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(
                f"Azure SQL table {self._qualified_table} is missing or does not "
                "have an embedding column"
            )
        dimension = int(_row_value(row, 0, "vector_dimensions"))
        base_type = int(_row_value(row, 1, "vector_base_type"))
        if dimension != self._dimension or base_type != 0:
            raise ValueError(
                f"Azure SQL table {self._qualified_table} uses VECTOR({dimension}) "
                f"base type {base_type}; expected float32 VECTOR({self._dimension})"
            )

    @contextmanager
    def _cursor(self, *, write: bool = False) -> Iterator[Any]:
        connection = self._connection_factory()
        cursor: Any | None = None
        try:
            cursor = connection.cursor()
            yield cursor
            if write:
                connection.commit()
        except BaseException:
            if write:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                connection.close()

    @staticmethod
    def _validate_identifier(value: str, label: str) -> None:
        if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
            raise ValueError(
                f"{label} must start with a letter or underscore and contain only "
                "letters, numbers, and underscores"
            )
        if len(value) > 128:
            raise ValueError(f"{label} cannot exceed 128 characters")

    @staticmethod
    def _validate_chunk(chunk: Chunk) -> None:
        if not isinstance(chunk.id, str) or not chunk.id:
            raise ValueError("chunk IDs must be non-empty strings")
        utf16_units = len(chunk.id.encode("utf-16-le")) // 2
        if utf16_units > 450:
            raise ValueError("Azure SQL chunk IDs cannot exceed 450 UTF-16 code units")
        if not isinstance(chunk.text, str):
            raise ValueError("chunk text must be a string")


def _mssql_connection_factory(connection_string: str) -> ConnectionFactory:
    def open_connection() -> Any:
        try:
            from mssql_python import connect
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AzureSqlVectorStore requires the Azure SQL extra; install it "
                "with `uv sync --extra azure-sql`"
            ) from exc
        return connect(connection_string)

    return open_connection


def _translate_filter(
    filter: MetadataFilter | None,
) -> tuple[str, list[MetadataValue]]:
    if not filter:
        return "", []

    clauses: list[str] = []
    parameters: list[MetadataValue] = []
    for key, condition in filter.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata filter keys must be non-empty strings")

        if not isinstance(condition, dict):
            predicate, equality_values = _equality_predicate(
                condition, "equality filter"
            )
            clauses.append(_metadata_exists(predicate))
            parameters.extend((key, *equality_values))
            continue
        if not condition:
            raise ValueError(f"metadata filter for {key!r} cannot be empty")

        for operator, expected in condition.items():
            if operator == "$in":
                if not isinstance(expected, (list, tuple, set, frozenset)):
                    raise ValueError("$in requires a list-like value")
                candidates = list(expected)
                if not candidates:
                    raise ValueError("$in requires at least one value")
                alternatives: list[str] = []
                membership_values: list[MetadataValue] = []
                for candidate in candidates:
                    predicate, predicate_values = _equality_predicate(
                        candidate, "$in filter"
                    )
                    alternatives.append(f"({predicate})")
                    membership_values.extend(predicate_values)
                clauses.append(_metadata_exists(" OR ".join(alternatives)))
                parameters.extend((key, *membership_values))
                continue

            sql_operator = _ORDERED_OPERATORS.get(operator)
            if sql_operator is None:
                raise ValueError(f"unsupported metadata filter operator: {operator!r}")
            if (
                isinstance(expected, bool)
                or not isinstance(expected, (int, float))
                or not _is_finite_number(expected)
            ):
                raise ValueError(f"{operator} requires a finite numeric value")
            clauses.append(
                _metadata_exists(
                    "metadata_item.[type] = 2 AND "
                    f"TRY_CONVERT(FLOAT, metadata_item.[value]) {sql_operator} ?"
                )
            )
            parameters.extend((key, expected))

    return " AND ".join(clauses), parameters


def _metadata_exists(value_predicate: str) -> str:
    return (
        "EXISTS (SELECT 1 FROM OPENJSON(records.[metadata_json]) "
        "AS metadata_item WHERE metadata_item.[key] "
        "COLLATE Latin1_General_100_BIN2 = ? AND "
        f"({value_predicate}))"
    )


def _equality_predicate(value: object, label: str) -> tuple[str, list[MetadataValue]]:
    if isinstance(value, bool):
        return (
            "metadata_item.[type] = 3 AND metadata_item.[value] "
            "COLLATE Latin1_General_100_BIN2 = ?",
            ["true" if value else "false"],
        )
    if isinstance(value, str):
        return (
            "metadata_item.[type] = 1 AND metadata_item.[value] "
            "COLLATE Latin1_General_100_BIN2 = ?",
            [value],
        )
    if isinstance(value, (int, float)) and _is_finite_number(value):
        return (
            "metadata_item.[type] = 2 AND "
            "TRY_CONVERT(FLOAT, metadata_item.[value]) = ?",
            [value],
        )
    raise ValueError(f"{label} requires a finite scalar metadata value")


def _serialize_metadata(metadata: dict[str, MetadataValue]) -> str:
    if not isinstance(metadata, dict):
        raise ValueError("chunk metadata must be a dictionary")
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        if isinstance(value, bool) or isinstance(value, (str, int)):
            continue
        if isinstance(value, float) and math.isfinite(value):
            continue
        raise ValueError("metadata values must be finite strings, numbers, or booleans")
    return json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_finite_number(value: float) -> bool:
    return isinstance(value, int) or math.isfinite(value)


def _search_result_from_row(row: Any) -> SearchResult:
    chunk = _chunk_from_row(row)
    score = float(_row_value(row, 3, "score"))
    return SearchResult(chunk=chunk, score=max(-1.0, min(1.0, score)))


def _chunk_from_row(row: Any) -> Chunk:
    id_ = str(_row_value(row, 0, "chunk_id"))
    text = str(_row_value(row, 1, "chunk_text"))
    raw_metadata = _row_value(row, 2, "metadata_json")
    try:
        metadata = json.loads(str(raw_metadata))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metadata JSON stored for chunk {id_!r}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata stored for chunk {id_!r} must be a JSON object")
    return Chunk(id=id_, text=text, metadata=metadata)


def _row_value(row: Any, position: int, name: str) -> Any:
    try:
        return row[position]
    except KeyError, IndexError, TypeError:
        try:
            return row[name]
        except KeyError, IndexError, TypeError:
            return getattr(row, name)


def _batches(values: Sequence[str], size: int = 1000) -> Iterator[Sequence[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _quote_identifier(identifier: str) -> str:
    return f"[{identifier}]"


def _primary_key_name(table_name: str) -> str:
    return f"PK_{table_name[:125]}"
