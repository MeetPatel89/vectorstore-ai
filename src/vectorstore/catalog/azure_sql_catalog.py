"""Azure SQL document catalog with native Full-Text Search.

The catalog stores structured documents, chunks, embedding lifecycle state,
and durable embedding-budget usage. Lexical retrieval is backed by a SQL
Server Full-Text catalog and ``CONTAINSTABLE`` ranking. Schema creation is an
explicit deployment operation; normal query and ingestion identities only
need DML permissions afterwards.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Self

from vectorstore.embeddings.base import EmbeddingSpec
from vectorstore.embeddings.policy import (
    BudgetReservation,
    BudgetReservationDecision,
    EmbeddingUsageRecord,
)
from vectorstore.embeddings.pricing import EmbeddingCharge, UsdAmount
from vectorstore.models import MetadataFilter, MetadataValue

from ._azure_sql_budget import _AzureSqlBudgetLedger
from ._azure_sql_support import (
    ConnectionFactory,
    _azure_sql_connection_factory,
    _AzureSqlDatabase,
    _iso_string,
    _optional_str,
    _row_value,
)
from .base import (
    CatalogChunk,
    CatalogDocument,
    EmbeddingState,
    LexicalUnavailableError,
    RankedHit,
    RetrievalScope,
)

Clock = Callable[[], datetime]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SEARCH_TERM = re.compile(r"[\w]+(?:[-.][\w]+)*", re.UNICODE)
_QUOTED_PHRASE = re.compile(r'"([^"]+)"')
_DOCUMENT_COLUMNS = frozenset(
    {
        "doc_id",
        "source",
        "title",
        "doc_type",
        "tenant_id",
        "visibility",
        "owner_group",
        "status",
        "created_at",
        "updated_at",
    }
)
_ORDERED_OPERATORS = {
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}
_UNRESTRICTED = RetrievalScope()
_SCHEMA_VERSION = 1
_AZURE_SQL_PARAMETER_BATCH = 1000
_LEXICAL_UNAVAILABLE_NUMBERS = frozenset(
    {
        207,  # invalid column name
        208,  # invalid object name
        7601,  # table is not full-text indexed
        7603,  # full-text catalog/index unavailable
        7645,  # full-text service/component unavailable
    }
)

_DOCUMENT_FIELDS = (
    "doc_id",
    "source",
    "title",
    "doc_type",
    "tenant_id",
    "visibility",
    "owner_group",
    "status",
    "created_at",
    "updated_at",
    "attributes_json",
)
_CHUNK_FIELDS = (
    "chunk_id",
    "doc_id",
    "chunk_index",
    "section_path",
    "text",
    "content_hash",
    "active",
)
_EMBEDDING_FIELDS = (
    "chunk_id",
    "space_id",
    "provider",
    "model",
    "dimension",
    "embedding_version",
    "content_hash",
    "created_at",
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AzureSqlDocumentCatalog:
    """Azure SQL implementation of ``DocumentCatalog`` and ``BudgetLedger``.

    Every public operation opens a fresh DB-API connection. Instances are
    therefore safe to share between request handlers without sharing cursors
    or transactions. ``mssql-python`` is imported lazily when an injected
    connection factory is not supplied.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        schema_name: str = "dbo",
        fulltext_catalog_name: str = "vectorstore_catalog_fulltext",
        language_lcid: int = 1033,
        initialize_schema: bool = False,
        connection_factory: ConnectionFactory | None = None,
        now: Clock = _utc_now,
    ) -> None:
        _validate_identifier(schema_name, "schema_name")
        _validate_identifier(fulltext_catalog_name, "fulltext_catalog_name")
        if (
            not isinstance(language_lcid, int)
            or isinstance(language_lcid, bool)
            or language_lcid <= 0
        ):
            raise ValueError("language_lcid must be a positive integer")
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
            connection_factory = _azure_sql_connection_factory(
                resolved_connection_string
            )
        elif not callable(connection_factory):
            raise TypeError("connection_factory must be callable")

        self._schema_name = schema_name
        self._fulltext_catalog_name = fulltext_catalog_name
        self._language_lcid = language_lcid
        self._database = _AzureSqlDatabase(connection_factory)
        self._now = now

        schema = _quote_identifier(schema_name)
        self._documents = f"{schema}.[documents]"
        self._chunks = f"{schema}.[chunks]"
        self._chunk_embeddings = f"{schema}.[chunk_embeddings]"
        self._embedding_usage = f"{schema}.[embedding_usage]"
        self._catalog_schema = f"{schema}.[catalog_schema]"
        self._chunks_primary_key = "PK_vectorstore_catalog_chunks"
        self._budget = _AzureSqlBudgetLedger(
            self._database,
            usage_table=self._embedding_usage,
            schema_table=self._catalog_schema,
            schema_version=_SCHEMA_VERSION,
            now=now,
        )

        if initialize_schema:
            self.create_schema()

    @property
    def schema_name(self) -> str:
        """The immutable Azure SQL schema configured for this catalog."""
        return self._schema_name

    @property
    def fulltext_catalog_name(self) -> str:
        """The database-scoped SQL Server Full-Text catalog name."""
        return self._fulltext_catalog_name

    @property
    def language_lcid(self) -> int:
        """The Full-Text word-breaker language identifier."""
        return self._language_lcid

    @property
    def schema_sql(self) -> str:
        """Versioned, idempotent DDL for the Azure SQL catalog."""
        schema = _quote_identifier(self.schema_name)
        schema_literal = _quote_literal(self.schema_name)
        fulltext_name = _quote_identifier(self.fulltext_catalog_name)
        fulltext_literal = _quote_literal(self.fulltext_catalog_name)
        return f"""
IF SCHEMA_ID(N{schema_literal}) IS NULL
    EXEC(N'CREATE SCHEMA {schema}');

IF OBJECT_ID(N'{self._catalog_schema}', N'U') IS NULL
BEGIN
    CREATE TABLE {self._catalog_schema} (
        [singleton] BIT NOT NULL,
        [version] INT NOT NULL,
        [fulltext_catalog_name] NVARCHAR(128) NOT NULL,
        [language_lcid] INT NOT NULL,
        CONSTRAINT [PK_vectorstore_catalog_schema]
            PRIMARY KEY CLUSTERED ([singleton]),
        CONSTRAINT [CK_vectorstore_catalog_schema_singleton]
            CHECK ([singleton] = 1),
        CONSTRAINT [CK_vectorstore_catalog_schema_version]
            CHECK ([version] > 0),
        CONSTRAINT [CK_vectorstore_catalog_schema_language]
            CHECK ([language_lcid] > 0)
    );
END;
IF NOT EXISTS (SELECT 1 FROM {self._catalog_schema} WHERE [singleton] = 1)
BEGIN
    INSERT INTO {self._catalog_schema}
        ([singleton], [version], [fulltext_catalog_name], [language_lcid])
    VALUES (1, {_SCHEMA_VERSION}, N{fulltext_literal}, {self.language_lcid});
END;

IF OBJECT_ID(N'{self._documents}', N'U') IS NULL
BEGIN
    CREATE TABLE {self._documents} (
        [doc_id] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NOT NULL,
        [source] NVARCHAR(MAX) COLLATE Latin1_General_100_BIN2 NULL,
        [title] NVARCHAR(MAX) COLLATE Latin1_General_100_BIN2 NULL,
        [doc_type] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NULL,
        [tenant_id] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NULL,
        [visibility] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NULL,
        [owner_group] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NULL,
        [status] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NULL,
        [created_at] NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NULL,
        [updated_at] NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NULL,
        [attributes_json] NVARCHAR(MAX) NOT NULL
            CONSTRAINT [DF_vectorstore_documents_attributes] DEFAULT N'{{}}',
        CONSTRAINT [PK_vectorstore_catalog_documents]
            PRIMARY KEY CLUSTERED ([doc_id]),
        CONSTRAINT [CK_vectorstore_documents_attributes_json]
            CHECK (ISJSON([attributes_json]) = 1)
    );
END;
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE [object_id] = OBJECT_ID(N'{self._documents}')
        AND [name] = N'IX_vectorstore_documents_doc_type'
)
    CREATE INDEX [IX_vectorstore_documents_doc_type]
        ON {self._documents} ([doc_type]);
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE [object_id] = OBJECT_ID(N'{self._documents}')
        AND [name] = N'IX_vectorstore_documents_tenant_id'
)
    CREATE INDEX [IX_vectorstore_documents_tenant_id]
        ON {self._documents} ([tenant_id]);
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE [object_id] = OBJECT_ID(N'{self._documents}')
        AND [name] = N'IX_vectorstore_documents_status'
)
    CREATE INDEX [IX_vectorstore_documents_status]
        ON {self._documents} ([status]);
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE [object_id] = OBJECT_ID(N'{self._documents}')
        AND [name] = N'IX_vectorstore_documents_updated_at'
)
    CREATE INDEX [IX_vectorstore_documents_updated_at]
        ON {self._documents} ([updated_at]);

IF OBJECT_ID(N'{self._chunks}', N'U') IS NULL
BEGIN
    CREATE TABLE {self._chunks} (
        [chunk_id] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NOT NULL,
        [doc_id] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NOT NULL,
        [chunk_index] INT NOT NULL CONSTRAINT [DF_vectorstore_chunks_index] DEFAULT 0,
        [section_path] NVARCHAR(MAX) NULL,
        [text] NVARCHAR(MAX) NOT NULL,
        [content_hash] NVARCHAR(128) NOT NULL,
        [active] BIT NOT NULL CONSTRAINT [DF_vectorstore_chunks_active] DEFAULT 1,
        CONSTRAINT [{self._chunks_primary_key}]
            PRIMARY KEY CLUSTERED ([chunk_id]),
        CONSTRAINT [FK_vectorstore_chunks_documents]
            FOREIGN KEY ([doc_id]) REFERENCES {self._documents} ([doc_id])
            ON DELETE CASCADE,
        CONSTRAINT [CK_vectorstore_chunks_index] CHECK ([chunk_index] >= 0)
    );
END;
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE [object_id] = OBJECT_ID(N'{self._chunks}')
        AND [name] = N'IX_vectorstore_chunks_doc_id'
)
    CREATE INDEX [IX_vectorstore_chunks_doc_id]
        ON {self._chunks} ([doc_id]);

IF OBJECT_ID(N'{self._chunk_embeddings}', N'U') IS NULL
BEGIN
    CREATE TABLE {self._chunk_embeddings} (
        [chunk_id] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NOT NULL,
        [space_id] NVARCHAR(300) COLLATE Latin1_General_100_BIN2 NOT NULL,
        [provider] NVARCHAR(450) NOT NULL,
        [model] NVARCHAR(450) NOT NULL,
        [dimension] INT NOT NULL,
        [embedding_version] NVARCHAR(450) NOT NULL,
        [content_hash] NVARCHAR(128) NOT NULL,
        [created_at] DATETIMEOFFSET(6) NOT NULL,
        CONSTRAINT [PK_vectorstore_chunk_embeddings]
            PRIMARY KEY NONCLUSTERED ([chunk_id], [space_id]),
        CONSTRAINT [FK_vectorstore_chunk_embeddings_chunks]
            FOREIGN KEY ([chunk_id]) REFERENCES {self._chunks} ([chunk_id])
            ON DELETE CASCADE,
        CONSTRAINT [CK_vectorstore_chunk_embeddings_dimension]
            CHECK ([dimension] > 0)
    );
END;

IF OBJECT_ID(N'{self._embedding_usage}', N'U') IS NULL
BEGIN
    CREATE TABLE {self._embedding_usage} (
        [event_id] CHAR(32) NOT NULL,
        [date] DATE NOT NULL,
        [provider] NVARCHAR(450) NOT NULL,
        [model] NVARCHAR(450) NOT NULL,
        [processing_mode] NVARCHAR(450) NOT NULL,
        [tokens] BIGINT NOT NULL,
        [rate_nanos_per_million] BIGINT NULL,
        [price_version] NVARCHAR(450) NULL,
        [charge_nanos] BIGINT NULL,
        [status] NVARCHAR(16) NOT NULL,
        [expires_at] DATETIMEOFFSET(6) NULL,
        [created_at] DATETIMEOFFSET(6) NOT NULL,
        [updated_at] DATETIMEOFFSET(6) NOT NULL,
        CONSTRAINT [PK_vectorstore_embedding_usage]
            PRIMARY KEY CLUSTERED ([event_id]),
        CONSTRAINT [CK_vectorstore_embedding_usage_tokens]
            CHECK ([tokens] >= 0),
        CONSTRAINT [CK_vectorstore_embedding_usage_rate]
            CHECK ([rate_nanos_per_million] IS NULL OR [rate_nanos_per_million] >= 0),
        CONSTRAINT [CK_vectorstore_embedding_usage_charge]
            CHECK ([charge_nanos] IS NULL OR [charge_nanos] >= 0),
        CONSTRAINT [CK_vectorstore_embedding_usage_status]
            CHECK ([status] IN (N'reserved', N'committed', N'released', N'expired')),
        CONSTRAINT [CK_vectorstore_embedding_usage_pricing]
            CHECK (
                ([rate_nanos_per_million] IS NULL
                    AND [price_version] IS NULL AND [charge_nanos] IS NULL)
                OR
                ([rate_nanos_per_million] IS NOT NULL
                    AND [price_version] IS NOT NULL AND [charge_nanos] IS NOT NULL)
            )
    );
END;
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE [object_id] = OBJECT_ID(N'{self._embedding_usage}')
        AND [name] = N'IX_vectorstore_embedding_usage_date'
)
    CREATE INDEX [IX_vectorstore_embedding_usage_date]
        ON {self._embedding_usage} ([date]);
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE [object_id] = OBJECT_ID(N'{self._embedding_usage}')
        AND [name] = N'IX_vectorstore_embedding_usage_status_date'
)
    CREATE INDEX [IX_vectorstore_embedding_usage_status_date]
        ON {self._embedding_usage} ([status], [date]);
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE [object_id] = OBJECT_ID(N'{self._embedding_usage}')
        AND [name] = N'IX_vectorstore_embedding_usage_provider_model_date'
)
    CREATE INDEX [IX_vectorstore_embedding_usage_provider_model_date]
        ON {self._embedding_usage} ([provider], [date]) INCLUDE ([model]);

IF COALESCE(FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'), 0) <> 1
    THROW 50001, 'SQL Server Full-Text Search is not installed or enabled.', 1;
IF NOT EXISTS (
    SELECT 1 FROM sys.fulltext_catalogs WHERE [name] = N{fulltext_literal}
)
    CREATE FULLTEXT CATALOG {fulltext_name};
IF NOT EXISTS (
    SELECT 1 FROM sys.fulltext_indexes
    WHERE [object_id] = OBJECT_ID(N'{self._chunks}')
)
    CREATE FULLTEXT INDEX ON {self._chunks}
        ([text] LANGUAGE {self.language_lcid})
        KEY INDEX [{self._chunks_primary_key}]
        ON {fulltext_name}
        WITH (CHANGE_TRACKING = AUTO, STOPLIST = OFF);
""".strip()

    def create_schema(self) -> None:
        """Create tables transactionally and Full-Text objects in autocommit."""
        core_sql, fulltext_sql = self._schema_statements
        with self._database.cursor(write=True) as cursor:
            cursor.execute(core_sql)
            self._validate_metadata_cursor(cursor)
        # SQL Server rejects CREATE FULLTEXT INDEX inside a user transaction.
        # A second autocommit connection also leaves the already-valid core
        # schema recoverable when optional Full-Text provisioning fails.
        with self._database.cursor(autocommit=True) as cursor:
            cursor.execute(fulltext_sql)
            self._validate_fulltext_cursor(cursor)

    def validate_schema(self) -> None:
        """Verify schema metadata and the chunks Full-Text index."""
        with self._database.cursor() as cursor:
            self._validate_schema_cursor(cursor)

    @property
    def _schema_statements(self) -> tuple[str, str]:
        marker = "IF COALESCE(FULLTEXTSERVICEPROPERTY"
        core_sql, separator, fulltext_tail = self.schema_sql.partition(marker)
        if not separator:  # pragma: no cover - developer invariant
            raise RuntimeError("Azure SQL Full-Text DDL marker is missing")
        return core_sql.rstrip(), f"{separator}{fulltext_tail}"

    def close(self) -> None:
        """Release catalog-owned resources (connections are operation-scoped)."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- structured documents and chunks ---------------------------------

    def upsert_documents(self, documents: list[CatalogDocument]) -> None:
        """Insert or update documents by document ID."""
        prepared: dict[str, tuple[object, ...]] = {}
        for document in documents:
            _validate_key(document.doc_id, "doc_id", 450)
            prepared[document.doc_id] = (
                document.source,
                document.title,
                document.doc_type,
                document.tenant_id,
                document.visibility,
                document.owner_group,
                document.status,
                document.created_at,
                document.updated_at,
                _serialize_attributes(document.attributes),
                document.doc_id,
                document.doc_id,
                document.source,
                document.title,
                document.doc_type,
                document.tenant_id,
                document.visibility,
                document.owner_group,
                document.status,
                document.created_at,
                document.updated_at,
                _serialize_attributes(document.attributes),
            )
        if not prepared:
            return
        statement = f"""
UPDATE {self._documents} WITH (UPDLOCK, SERIALIZABLE)
SET [source] = ?, [title] = ?, [doc_type] = ?, [tenant_id] = ?,
    [visibility] = ?, [owner_group] = ?, [status] = ?, [created_at] = ?,
    [updated_at] = ?, [attributes_json] = ?
WHERE [doc_id] = ?;
IF @@ROWCOUNT = 0
BEGIN
    INSERT INTO {self._documents} (
        [doc_id], [source], [title], [doc_type], [tenant_id], [visibility],
        [owner_group], [status], [created_at], [updated_at], [attributes_json]
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
END;
""".strip()
        with self._database.cursor(write=True) as cursor:
            for parameters in prepared.values():
                cursor.execute(statement, parameters)

    def upsert_chunks(self, chunks: list[CatalogChunk]) -> None:
        """Insert or update chunks; Full-Text change tracking refreshes the index."""
        prepared = _prepare_chunks(chunks)
        if not prepared:
            return
        statement = self._chunk_upsert_sql
        with self._database.cursor(write=True) as cursor:
            for parameters in prepared.values():
                cursor.execute(statement, parameters)

    def replace_chunks(self, doc_id: str, chunks: list[CatalogChunk]) -> list[str]:
        """Replace one document's chunks in one Azure SQL transaction."""
        _validate_key(doc_id, "doc_id", 450)
        if any(chunk.doc_id != doc_id for chunk in chunks):
            raise ValueError("every replacement chunk must belong to doc_id")
        new_ids = {chunk.chunk_id for chunk in chunks}
        if len(new_ids) != len(chunks):
            raise ValueError("replacement chunk IDs must be unique")
        prepared = _prepare_chunks(chunks)

        with self._database.cursor(write=True) as cursor:
            cursor.execute(
                f"SELECT [chunk_id] FROM {self._chunks} WITH (UPDLOCK, HOLDLOCK) "
                "WHERE [doc_id] = ?",
                (doc_id,),
            )
            old_ids = {str(_row_value(row, 0, "chunk_id")) for row in cursor.fetchall()}
            for parameters in prepared.values():
                cursor.execute(self._chunk_upsert_sql, parameters)
            removed = sorted(old_ids - new_ids)
            for batch in _batches(removed):
                cursor.execute(
                    f"DELETE FROM {self._chunks} "
                    f"WHERE [chunk_id] IN ({_placeholders(len(batch))})",
                    tuple(batch),
                )
        return removed

    def delete_documents(self, doc_ids: list[str]) -> None:
        """Delete documents; foreign keys cascade chunks and lifecycle state."""
        unique_ids = list(dict.fromkeys(doc_ids))
        for doc_id in unique_ids:
            _validate_key(doc_id, "doc_id", 450)
        if not unique_ids:
            return
        with self._database.cursor(write=True) as cursor:
            for batch in _batches(unique_ids):
                cursor.execute(
                    f"DELETE FROM {self._documents} "
                    f"WHERE [doc_id] IN ({_placeholders(len(batch))})",
                    tuple(batch),
                )

    @property
    def _chunk_upsert_sql(self) -> str:
        return f"""
UPDATE {self._chunks} WITH (UPDLOCK, SERIALIZABLE)
SET [doc_id] = ?, [chunk_index] = ?, [section_path] = ?, [text] = ?,
    [content_hash] = ?, [active] = ?
WHERE [chunk_id] = ?;
IF @@ROWCOUNT = 0
BEGIN
    INSERT INTO {self._chunks} (
        [chunk_id], [doc_id], [chunk_index], [section_path], [text],
        [content_hash], [active]
    ) VALUES (?, ?, ?, ?, ?, ?, ?);
END;
""".strip()

    def find(
        self,
        filter: MetadataFilter | None = None,
        scope: RetrievalScope | None = None,
        limit: int = 100,
    ) -> list[CatalogDocument]:
        """Find documents with filters and authorization scope pushed into SQL."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        clauses, parameters = _scope_clauses(scope or _UNRESTRICTED)
        if filter:
            filter_clauses, filter_parameters = _filter_clauses(filter)
            clauses.extend(filter_clauses)
            parameters.extend(filter_parameters)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        fields = ", ".join(f"d.[{field}]" for field in _DOCUMENT_FIELDS)
        statement = (
            f"SELECT TOP (?) {fields} FROM {self._documents} AS d "
            f"{where} ORDER BY d.[doc_id]"
        )
        with self._database.cursor() as cursor:
            cursor.execute(statement, (limit, *parameters))
            rows = cursor.fetchall()
        return [_document_from_row(row) for row in rows]

    def get_chunks(self, chunk_ids: list[str]) -> list[CatalogChunk]:
        """Return known chunks in requested-ID order."""
        if not chunk_ids:
            return []
        unique_ids = list(dict.fromkeys(chunk_ids))
        for chunk_id in unique_ids:
            _validate_key(chunk_id, "chunk_id", 450)
        fields = ", ".join(f"[{field}]" for field in _CHUNK_FIELDS)
        by_id: dict[str, CatalogChunk] = {}
        with self._database.cursor() as cursor:
            for batch in _batches(unique_ids):
                cursor.execute(
                    f"SELECT {fields} FROM {self._chunks} "
                    f"WHERE [chunk_id] IN ({_placeholders(len(batch))})",
                    tuple(batch),
                )
                for row in cursor.fetchall():
                    chunk = _chunk_from_row(row)
                    by_id[chunk.chunk_id] = chunk
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    # -- lexical retrieval -------------------------------------------------

    def search_lexical(
        self,
        query: str,
        k: int = 10,
        filter: MetadataFilter | None = None,
        scope: RetrievalScope | None = None,
    ) -> list[RankedHit]:
        """Rank active chunks with Azure SQL ``CONTAINSTABLE`` Full-Text Search."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("lexical query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be greater than zero")
        search_condition = _contains_search_condition(query)

        clauses, parameters = _scope_clauses(scope or _UNRESTRICTED)
        if filter:
            filter_clauses, filter_parameters = _filter_clauses(filter)
            clauses.extend(filter_clauses)
            parameters.extend(filter_parameters)
        extra = f" AND {' AND '.join(clauses)}" if clauses else ""
        statement = f"""
SELECT TOP (?)
    c.[chunk_id],
    CAST(fulltext_result.[RANK] AS FLOAT) AS [score]
FROM CONTAINSTABLE(
    {self._chunks}, ([text]), ?, LANGUAGE {self.language_lcid}
) AS fulltext_result
JOIN {self._chunks} AS c ON c.[chunk_id] = fulltext_result.[KEY]
JOIN {self._documents} AS d ON d.[doc_id] = c.[doc_id]
WHERE c.[active] = 1{extra}
ORDER BY fulltext_result.[RANK] DESC, c.[chunk_id] ASC
""".strip()
        try:
            with self._database.cursor() as cursor:
                cursor.execute(statement, (k, search_condition, *parameters))
                rows = cursor.fetchall()
        except Exception as exc:
            if _is_lexical_unavailable(exc):
                raise LexicalUnavailableError(
                    f"Azure SQL lexical search is unavailable: {exc}"
                ) from exc
            raise

        return [
            RankedHit(
                chunk_id=str(_row_value(row, 0, "chunk_id")),
                rank=position,
                score=float(_row_value(row, 1, "score")),
            )
            for position, row in enumerate(rows, start=1)
        ]

    # -- embedding lifecycle ledger --------------------------------------

    def embedding_state(
        self, space_id: str, chunk_ids: list[str] | None = None
    ) -> dict[str, EmbeddingState]:
        """Return recorded embedding state for one embedding space."""
        _validate_key(space_id, "space_id", 300)
        fields = ", ".join(f"[{field}]" for field in _EMBEDDING_FIELDS)
        states: list[EmbeddingState] = []
        if chunk_ids is None:
            with self._database.cursor() as cursor:
                cursor.execute(
                    f"SELECT {fields} FROM {self._chunk_embeddings} "
                    "WHERE [space_id] = ?",
                    (space_id,),
                )
                states.extend(
                    _embedding_state_from_row(row) for row in cursor.fetchall()
                )
        else:
            if not chunk_ids:
                return {}
            unique_ids = list(dict.fromkeys(chunk_ids))
            for chunk_id in unique_ids:
                _validate_key(chunk_id, "chunk_id", 450)
            with self._database.cursor() as cursor:
                for batch in _batches(unique_ids):
                    cursor.execute(
                        f"SELECT {fields} FROM {self._chunk_embeddings} "
                        f"WHERE [space_id] = ? AND [chunk_id] IN "
                        f"({_placeholders(len(batch))})",
                        (space_id, *batch),
                    )
                    states.extend(
                        _embedding_state_from_row(row) for row in cursor.fetchall()
                    )
        return {state.chunk_id: state for state in states}

    def mark_embedded(
        self, chunk_id: str, spec: EmbeddingSpec, content_hash: str
    ) -> None:
        """Record that a current vector exists for one chunk and space."""
        _validate_key(chunk_id, "chunk_id", 450)
        _validate_key(spec.space_id, "space_id", 300)
        _validate_key(content_hash, "content_hash", 128)
        statement = f"""
UPDATE {self._chunk_embeddings} WITH (UPDLOCK, SERIALIZABLE)
SET [provider] = ?, [model] = ?, [dimension] = ?, [embedding_version] = ?,
    [content_hash] = ?, [created_at] = ?
WHERE [chunk_id] = ? AND [space_id] = ?;
IF @@ROWCOUNT = 0
BEGIN
    INSERT INTO {self._chunk_embeddings} (
        [chunk_id], [space_id], [provider], [model], [dimension],
        [embedding_version], [content_hash], [created_at]
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
END;
""".strip()
        created_at = self._now()
        with self._database.cursor(write=True) as cursor:
            cursor.execute(
                statement,
                (
                    spec.provider,
                    spec.model,
                    spec.dimension,
                    spec.version,
                    content_hash,
                    created_at,
                    chunk_id,
                    spec.space_id,
                    chunk_id,
                    spec.space_id,
                    spec.provider,
                    spec.model,
                    spec.dimension,
                    spec.version,
                    content_hash,
                    created_at,
                ),
            )

    def invalidate_embeddings(self, chunk_ids: list[str]) -> None:
        """Remove lifecycle state after authorization/filter metadata changes."""
        unique_ids = list(dict.fromkeys(chunk_ids))
        for chunk_id in unique_ids:
            _validate_key(chunk_id, "chunk_id", 450)
        if not unique_ids:
            return
        with self._database.cursor(write=True) as cursor:
            for batch in _batches(unique_ids):
                cursor.execute(
                    f"DELETE FROM {self._chunk_embeddings} "
                    f"WHERE [chunk_id] IN ({_placeholders(len(batch))})",
                    tuple(batch),
                )

    def stale_chunk_ids(self, spec: EmbeddingSpec) -> list[str]:
        """Return active chunks missing a current vector for the space."""
        _validate_key(spec.space_id, "space_id", 300)
        statement = f"""
SELECT c.[chunk_id]
FROM {self._chunks} AS c
LEFT JOIN {self._chunk_embeddings} AS e
    ON e.[chunk_id] = c.[chunk_id] AND e.[space_id] = ?
WHERE c.[active] = 1
    AND (e.[chunk_id] IS NULL OR e.[content_hash] <> c.[content_hash])
ORDER BY c.[chunk_id]
""".strip()
        with self._database.cursor() as cursor:
            cursor.execute(statement, (spec.space_id,))
            rows = cursor.fetchall()
        return [str(_row_value(row, 0, "chunk_id")) for row in rows]

    # -- durable budget facade --------------------------------------------

    def reserve(
        self,
        charge: EmbeddingCharge,
        *,
        daily_limit_nanos: int | None,
        monthly_limit_nanos: int | None,
        ttl_seconds: float,
    ) -> BudgetReservationDecision:
        """Delegate atomic spend reservation to the Azure SQL ledger."""
        return self._budget.reserve(
            charge,
            daily_limit_nanos=daily_limit_nanos,
            monthly_limit_nanos=monthly_limit_nanos,
            ttl_seconds=ttl_seconds,
        )

    def commit(
        self,
        reservation: BudgetReservation,
        actual_charge: EmbeddingCharge,
    ) -> None:
        """Delegate reservation reconciliation to the Azure SQL ledger."""
        self._budget.commit(reservation, actual_charge)

    def release(self, reservation: BudgetReservation) -> None:
        """Delegate reservation release to the Azure SQL ledger."""
        self._budget.release(reservation)

    def record(
        self,
        charge_or_provider: EmbeddingCharge | str,
        tokens: int | None = None,
        usd: UsdAmount | None = None,
        *,
        model: str = "<unspecified>",
        processing_mode: str = "standard",
        price_version: str = "legacy-explicit-total",
    ) -> None:
        """Delegate committed usage recording to the Azure SQL ledger."""
        self._budget.record(
            charge_or_provider,
            tokens,
            usd,
            model=model,
            processing_mode=processing_mode,
            price_version=price_version,
        )

    def spent_today_nanos(self) -> int:
        """Return committed plus reserved nanodollars for today."""
        return self._budget.spent_today_nanos()

    def spent_month_nanos(self) -> int:
        """Return committed plus reserved nanodollars for this month."""
        return self._budget.spent_month_nanos()

    def spent_today(self) -> Decimal:
        """Return exact committed plus reserved USD spend for today."""
        return self._budget.spent_today()

    def spent_month(self) -> Decimal:
        """Return exact committed plus reserved USD spend for this month."""
        return self._budget.spent_month()

    def tokens_today(self, provider: str, model: str | None = None) -> int:
        """Return committed tokens today, optionally filtered by model."""
        return self._budget.tokens_today(provider, model)

    def usage_records(self) -> tuple[EmbeddingUsageRecord, ...]:
        """Return all usage and reservation audit records."""
        return self._budget.usage_records()

    def _validate_schema_cursor(self, cursor: Any) -> None:
        self._validate_metadata_cursor(cursor)
        self._validate_fulltext_cursor(cursor)

    def _validate_metadata_cursor(self, cursor: Any) -> None:
        cursor.execute(
            f"SELECT [version], [fulltext_catalog_name], [language_lcid] "
            f"FROM {self._catalog_schema} WHERE [singleton] = 1"
        )
        metadata_row = cursor.fetchone()
        if metadata_row is None:
            raise RuntimeError("Azure SQL catalog schema metadata is missing")
        version = int(_row_value(metadata_row, 0, "version"))
        configured_catalog = str(_row_value(metadata_row, 1, "fulltext_catalog_name"))
        configured_language = int(_row_value(metadata_row, 2, "language_lcid"))
        if version != _SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported Azure SQL catalog schema version {version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if configured_catalog != self.fulltext_catalog_name:
            raise ValueError(
                "Azure SQL Full-Text catalog mismatch: schema uses "
                f"{configured_catalog!r}, catalog requested "
                f"{self.fulltext_catalog_name!r}"
            )
        if configured_language != self.language_lcid:
            raise ValueError(
                "Azure SQL Full-Text language mismatch: schema uses "
                f"{configured_language}, catalog requested {self.language_lcid}"
            )

    def _validate_fulltext_cursor(self, cursor: Any) -> None:
        cursor.execute(
            """
SELECT fulltext_catalog.[name], fulltext_index.[change_tracking_state_desc],
    key_index.[name], fulltext_column.[language_id]
FROM sys.fulltext_indexes AS fulltext_index
JOIN sys.fulltext_catalogs AS fulltext_catalog
    ON fulltext_catalog.[fulltext_catalog_id] = fulltext_index.[fulltext_catalog_id]
JOIN sys.indexes AS key_index
    ON key_index.[object_id] = fulltext_index.[object_id]
    AND key_index.[index_id] = fulltext_index.[unique_index_id]
JOIN sys.fulltext_index_columns AS fulltext_column
    ON fulltext_column.[object_id] = fulltext_index.[object_id]
    AND fulltext_column.[column_id] = COLUMNPROPERTY(
        fulltext_index.[object_id], N'text', 'ColumnId'
    )
WHERE fulltext_index.[object_id] = OBJECT_ID(?)
""".strip(),
            (self._chunks,),
        )
        index_row = cursor.fetchone()
        if index_row is None:
            raise RuntimeError(
                f"Azure SQL table {self._chunks} is missing its Full-Text index"
            )
        index_catalog = str(_row_value(index_row, 0, "name"))
        tracking = str(_row_value(index_row, 1, "change_tracking_state_desc"))
        key_index = str(_row_value(index_row, 2, "key_index"))
        language_lcid = int(_row_value(index_row, 3, "language_id"))
        if index_catalog != self.fulltext_catalog_name:
            raise ValueError(
                f"Azure SQL table {self._chunks} uses Full-Text catalog "
                f"{index_catalog!r}; expected {self.fulltext_catalog_name!r}"
            )
        if tracking.upper() != "AUTO":
            raise ValueError(
                f"Azure SQL table {self._chunks} must use AUTO Full-Text "
                "change tracking"
            )
        if key_index != self._chunks_primary_key:
            raise ValueError(
                f"Azure SQL table {self._chunks} Full-Text key index is "
                f"{key_index!r}; expected {self._chunks_primary_key!r}"
            )
        if language_lcid != self.language_lcid:
            raise ValueError(
                f"Azure SQL table {self._chunks} Full-Text column uses language "
                f"{language_lcid}; expected {self.language_lcid}"
            )


# Short alias for callers that name catalogs by database rather than protocol.
AzureSqlCatalog = AzureSqlDocumentCatalog


def _prepare_chunks(chunks: list[CatalogChunk]) -> dict[str, tuple[object, ...]]:
    prepared: dict[str, tuple[object, ...]] = {}
    for chunk in chunks:
        _validate_key(chunk.chunk_id, "chunk_id", 450)
        _validate_key(chunk.doc_id, "doc_id", 450)
        assert chunk.content_hash is not None
        _validate_key(chunk.content_hash, "content_hash", 128)
        prepared[chunk.chunk_id] = (
            chunk.doc_id,
            chunk.chunk_index,
            chunk.section_path,
            chunk.text,
            chunk.content_hash,
            chunk.active,
            chunk.chunk_id,
            chunk.chunk_id,
            chunk.doc_id,
            chunk.chunk_index,
            chunk.section_path,
            chunk.text,
            chunk.content_hash,
            chunk.active,
        )
    return prepared


def _filter_clauses(filter: MetadataFilter) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    parameters: list[object] = []
    for key, condition in filter.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata filter keys must be non-empty strings")
        if key in _DOCUMENT_COLUMNS:
            _append_column_filter(clauses, parameters, key, condition)
        else:
            _append_json_filter(clauses, parameters, key, condition)
    return clauses, parameters


def _append_column_filter(
    clauses: list[str],
    parameters: list[object],
    key: str,
    condition: object,
) -> None:
    expression = f"d.[{key}]"
    if not isinstance(condition, dict):
        clauses.append(f"{expression} = ?")
        parameters.append(_text_filter_value(condition, "equality filter"))
        return
    if not condition:
        raise ValueError(f"metadata filter for {key!r} cannot be empty")
    for operator, expected in condition.items():
        if operator == "$in":
            candidates = _membership_values(expected)
            values = [
                _text_filter_value(candidate, "$in filter") for candidate in candidates
            ]
            clauses.append(f"{expression} IN ({_placeholders(len(values))})")
            parameters.extend(values)
            continue
        sql_operator = _ORDERED_OPERATORS.get(operator)
        if sql_operator is None:
            raise ValueError(f"unsupported metadata filter operator: {operator!r}")
        clauses.append(f"{expression} {sql_operator} ?")
        parameters.append(_text_filter_value(expected, f"{operator} filter"))


def _append_json_filter(
    clauses: list[str],
    parameters: list[object],
    key: str,
    condition: object,
) -> None:
    if not isinstance(condition, dict):
        predicate, equality_values = _json_equality_predicate(
            condition, "equality filter"
        )
        clauses.append(_attribute_exists(predicate))
        parameters.extend((key, *equality_values))
        return
    if not condition:
        raise ValueError(f"metadata filter for {key!r} cannot be empty")
    for operator, expected in condition.items():
        if operator == "$in":
            candidates = _membership_values(expected)
            alternatives: list[str] = []
            values: list[MetadataValue] = []
            for candidate in candidates:
                predicate, predicate_values = _json_equality_predicate(
                    candidate, "$in filter"
                )
                alternatives.append(f"({predicate})")
                values.extend(predicate_values)
            clauses.append(_attribute_exists(" OR ".join(alternatives)))
            parameters.extend((key, *values))
            continue
        sql_operator = _ORDERED_OPERATORS.get(operator)
        if sql_operator is None:
            raise ValueError(f"unsupported metadata filter operator: {operator!r}")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or (isinstance(expected, float) and not math.isfinite(expected))
        ):
            raise ValueError(f"{operator} requires a finite numeric value")
        clauses.append(
            _attribute_exists(
                "attribute_item.[type] = 2 AND "
                f"TRY_CONVERT(FLOAT, attribute_item.[value]) {sql_operator} ?"
            )
        )
        parameters.extend((key, expected))


def _attribute_exists(value_predicate: str) -> str:
    return (
        "EXISTS (SELECT 1 FROM OPENJSON(d.[attributes_json]) AS attribute_item "
        "WHERE attribute_item.[key] COLLATE Latin1_General_100_BIN2 = ? AND "
        f"({value_predicate}))"
    )


def _json_equality_predicate(
    value: object, label: str
) -> tuple[str, list[MetadataValue]]:
    if isinstance(value, bool):
        return (
            "attribute_item.[type] = 3 AND attribute_item.[value] "
            "COLLATE Latin1_General_100_BIN2 = ?",
            ["true" if value else "false"],
        )
    if isinstance(value, str):
        return (
            "attribute_item.[type] = 1 AND attribute_item.[value] "
            "COLLATE Latin1_General_100_BIN2 = ?",
            [value],
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} requires a finite scalar metadata value")
        return (
            "attribute_item.[type] = 2 AND "
            "TRY_CONVERT(FLOAT, attribute_item.[value]) = ?",
            [value],
        )
    raise ValueError(f"{label} requires a finite scalar metadata value")


def _scope_clauses(scope: RetrievalScope) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    parameters: list[object] = []
    if scope.tenant_id is not None:
        clauses.append("(d.[tenant_id] IS NULL OR d.[tenant_id] = ?)")
        parameters.append(scope.tenant_id)
    if scope.visibility is not None:
        clauses.append(
            "(d.[visibility] IS NULL OR "
            f"d.[visibility] IN ({_placeholders(len(scope.visibility))}))"
        )
        parameters.extend(scope.visibility)
    return clauses, parameters


def _membership_values(value: object) -> list[object]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError("$in requires a list-like value")
    values = list(value)
    if not values:
        raise ValueError("$in requires at least one value")
    return values


def _text_filter_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} on a document column requires a string")
    return value


def _serialize_attributes(attributes: Mapping[str, MetadataValue]) -> str:
    for key, value in attributes.items():
        if not isinstance(key, str):
            raise ValueError("attribute keys must be strings")
        _json_equality_predicate(value, "document attribute")
    return json.dumps(
        dict(attributes),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _document_from_row(row: Any) -> CatalogDocument:
    return CatalogDocument(
        doc_id=str(_row_value(row, 0, "doc_id")),
        source=_optional_str(_row_value(row, 1, "source")),
        title=_optional_str(_row_value(row, 2, "title")),
        doc_type=_optional_str(_row_value(row, 3, "doc_type")),
        tenant_id=_optional_str(_row_value(row, 4, "tenant_id")),
        visibility=_optional_str(_row_value(row, 5, "visibility")),
        owner_group=_optional_str(_row_value(row, 6, "owner_group")),
        status=_optional_str(_row_value(row, 7, "status")),
        created_at=_optional_str(_row_value(row, 8, "created_at")),
        updated_at=_optional_str(_row_value(row, 9, "updated_at")),
        attributes=_attributes_from_value(_row_value(row, 10, "attributes_json")),
    )


def _chunk_from_row(row: Any) -> CatalogChunk:
    return CatalogChunk(
        chunk_id=str(_row_value(row, 0, "chunk_id")),
        doc_id=str(_row_value(row, 1, "doc_id")),
        chunk_index=int(_row_value(row, 2, "chunk_index")),
        section_path=_optional_str(_row_value(row, 3, "section_path")),
        text=str(_row_value(row, 4, "text")),
        content_hash=str(_row_value(row, 5, "content_hash")),
        active=_bool_from_value(_row_value(row, 6, "active")),
    )


def _embedding_state_from_row(row: Any) -> EmbeddingState:
    return EmbeddingState(
        chunk_id=str(_row_value(row, 0, "chunk_id")),
        space_id=str(_row_value(row, 1, "space_id")),
        provider=str(_row_value(row, 2, "provider")),
        model=str(_row_value(row, 3, "model")),
        dimension=int(_row_value(row, 4, "dimension")),
        version=str(_row_value(row, 5, "embedding_version")),
        content_hash=str(_row_value(row, 6, "content_hash")),
        created_at=_iso_string(_row_value(row, 7, "created_at")),
    )


def _attributes_from_value(value: object) -> dict[str, MetadataValue]:
    if isinstance(value, Mapping):
        decoded: object = dict(value)
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid attributes JSON stored for document") from exc
    else:
        raise ValueError("document attributes must be returned as JSON or a mapping")
    if not isinstance(decoded, dict):
        raise ValueError("document attributes JSON must contain an object")
    return decoded


def _contains_search_condition(query: str) -> str:
    phrases = [phrase.strip() for phrase in _QUOTED_PHRASE.findall(query)]
    unquoted = _QUOTED_PHRASE.sub(" ", query)
    terms = [*phrases, *_SEARCH_TERM.findall(unquoted)]
    unique_terms = list(dict.fromkeys(term for term in terms if term))
    if not unique_terms:
        raise ValueError("lexical query must contain at least one searchable term")
    escaped = (term.replace('"', '""') for term in unique_terms)
    return " AND ".join(f'"{term}"' for term in escaped)


def _is_lexical_unavailable(exc: Exception) -> bool:
    if getattr(exc, "sqlstate", None) in {"42S02", "42S22"}:
        return True
    for attribute in ("number", "errno", "code"):
        value = getattr(exc, attribute, None)
        try:
            if value is not None and int(value) in _LEXICAL_UNAVAILABLE_NUMBERS:
                return True
        except TypeError, ValueError:
            pass
    message = " ".join(str(value) for value in getattr(exc, "args", (exc,)))
    if any(f"({number})" in message for number in _LEXICAL_UNAVAILABLE_NUMBERS):
        return True
    lowered = message.casefold()
    if "invalid object name" in lowered or "invalid column name" in lowered:
        return True
    return "full-text" in lowered and any(
        marker in lowered
        for marker in ("not installed", "not enabled", "not indexed", "unavailable")
    )


def _bool_from_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true"}:
            return True
        if normalized in {"0", "false"}:
            return False
    raise ValueError("stored BIT value must be boolean-compatible")


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{label} must start with a letter or underscore and contain only "
            "letters, numbers, and underscores"
        )
    if len(value) > 128:
        raise ValueError(f"{label} cannot exceed 128 characters")


def _validate_key(value: str, label: str, max_utf16_units: int) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    units = len(value.encode("utf-16-le")) // 2
    if units > max_utf16_units:
        raise ValueError(
            f"Azure SQL {label} cannot exceed {max_utf16_units} UTF-16 code units"
        )


def _quote_identifier(identifier: str) -> str:
    return f"[{identifier}]"


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _batches(
    values: Sequence[str], size: int = _AZURE_SQL_PARAMETER_BATCH
) -> Iterator[Sequence[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]
