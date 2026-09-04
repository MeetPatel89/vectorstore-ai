"""PostgreSQL-backed document catalog with native full-text search.

The catalog stores structured documents, chunks, embedding lifecycle state,
and durable budget usage. Lexical retrieval uses a stored tsvector generated
from chunk text and a GIN inverted index. Raw queries are converted with
websearch_to_tsquery and ranked with ts_rank_cd.

Schema creation is explicit by default. Runtime callers therefore only need
DML privileges after a deployment identity has called create_schema.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping
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

from ._postgres_budget import _PostgresBudgetLedger
from ._postgres_support import (
    ConnectionFactory,
    _iso_string,
    _optional_str,
    _postgres_connection_factory,
    _PostgresDatabase,
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
_TEXT_SEARCH_CONFIG = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*$"
)
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
_LEXICAL_UNAVAILABLE_SQLSTATES = frozenset(
    {
        "3F000",  # invalid_schema_name
        "42P01",  # undefined_table
        "42703",  # undefined_column
        "42883",  # undefined_function
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


class PostgresDocumentCatalog:
    """A PostgreSQL implementation of DocumentCatalog and BudgetLedger.

    A fresh DB-API connection is opened for every public operation, so an
    instance can safely be shared between request handlers. Pass a custom
    connection_factory for an application-owned pool or for tests. Otherwise
    Psycopg 3 is loaded lazily from the optional postgres dependency.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        schema_name: str = "public",
        text_search_config: str = "simple",
        initialize_schema: bool = False,
        connection_factory: ConnectionFactory | None = None,
        now: Clock = _utc_now,
    ) -> None:
        _validate_identifier(schema_name, "schema_name")
        _validate_text_search_config(text_search_config)
        if connection_factory is not None and connection_string is not None:
            raise ValueError(
                "pass either connection_string or connection_factory, not both"
            )
        if connection_factory is None:
            resolved_connection_string = connection_string or os.environ.get(
                "POSTGRES_CONNECTIONSTRING"
            )
            if not resolved_connection_string:
                raise ValueError(
                    "PostgreSQL connection string is required; pass "
                    "connection_string or set POSTGRES_CONNECTIONSTRING"
                )
            connection_factory = _postgres_connection_factory(
                resolved_connection_string
            )
        elif not callable(connection_factory):
            raise TypeError("connection_factory must be callable")

        self._schema_name = schema_name
        self._text_search_config = text_search_config
        self._database = _PostgresDatabase(connection_factory)
        self._now = now

        quoted_schema = _quote_identifier(schema_name)
        self._documents = f"{quoted_schema}.documents"
        self._chunks = f"{quoted_schema}.chunks"
        self._chunk_embeddings = f"{quoted_schema}.chunk_embeddings"
        self._embedding_usage = f"{quoted_schema}.embedding_usage"
        self._catalog_schema = f"{quoted_schema}.catalog_schema"
        self._text_search_config_sql = _quote_literal(text_search_config)
        self._budget = _PostgresBudgetLedger(
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
        """The immutable PostgreSQL schema configured for this catalog."""
        return self._schema_name

    @property
    def text_search_config(self) -> str:
        """The immutable PostgreSQL text-search configuration."""
        return self._text_search_config

    @property
    def schema_sql(self) -> str:
        """Idempotent PostgreSQL DDL for the catalog schema."""
        schema = _quote_identifier(self.schema_name)
        config = self._text_search_config_sql
        return f"""
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {self._catalog_schema} (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    version INTEGER NOT NULL CHECK (version > 0),
    text_search_config TEXT NOT NULL
);
INSERT INTO {self._catalog_schema} (singleton, version, text_search_config)
VALUES (TRUE, {_SCHEMA_VERSION}, {config})
ON CONFLICT (singleton) DO NOTHING;

CREATE TABLE IF NOT EXISTS {self._documents} (
    doc_id TEXT PRIMARY KEY,
    source TEXT,
    title TEXT,
    doc_type TEXT,
    tenant_id TEXT,
    visibility TEXT,
    owner_group TEXT,
    status TEXT,
    created_at TEXT,
    updated_at TEXT,
    attributes_json JSONB NOT NULL DEFAULT '{{}}'::jsonb
        CHECK (jsonb_typeof(attributes_json) = 'object')
);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type
    ON {self._documents} (doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_tenant_id
    ON {self._documents} (tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_status
    ON {self._documents} (status);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at
    ON {self._documents} (updated_at);

CREATE TABLE IF NOT EXISTS {self._chunks} (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES {self._documents} (doc_id)
        ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    section_path TEXT,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector({config}::regconfig, text)
    ) STORED
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON {self._chunks} (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
    ON {self._chunks} USING GIN (search_vector);

CREATE TABLE IF NOT EXISTS {self._chunk_embeddings} (
    chunk_id TEXT NOT NULL REFERENCES {self._chunks} (chunk_id)
        ON DELETE CASCADE,
    space_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    embedding_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (chunk_id, space_id)
);

CREATE TABLE IF NOT EXISTS {self._embedding_usage} (
    event_id TEXT PRIMARY KEY,
    date DATE NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    processing_mode TEXT NOT NULL,
    tokens BIGINT NOT NULL CHECK (tokens >= 0),
    rate_nanos_per_million BIGINT CHECK (rate_nanos_per_million >= 0),
    price_version TEXT,
    charge_nanos BIGINT CHECK (charge_nanos >= 0),
    status TEXT NOT NULL CHECK (
        status IN ('reserved', 'committed', 'released', 'expired')
    ),
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CHECK (
        (
            rate_nanos_per_million IS NULL
            AND price_version IS NULL
            AND charge_nanos IS NULL
        )
        OR
        (
            rate_nanos_per_million IS NOT NULL
            AND price_version IS NOT NULL
            AND charge_nanos IS NOT NULL
        )
    )
);
CREATE INDEX IF NOT EXISTS idx_embedding_usage_date
    ON {self._embedding_usage} (date);
CREATE INDEX IF NOT EXISTS idx_embedding_usage_status_date
    ON {self._embedding_usage} (status, date);
CREATE INDEX IF NOT EXISTS idx_embedding_usage_provider_model_date
    ON {self._embedding_usage} (provider, model, date);
""".strip()

    def create_schema(self) -> None:
        """Create the catalog schema and validate its lexical index."""
        with self._database.cursor(write=True) as cursor:
            cursor.execute(self.schema_sql)
            self._validate_schema_cursor(cursor)

    def validate_schema(self) -> None:
        """Verify the generated tsvector column and its GIN index."""
        with self._database.cursor() as cursor:
            self._validate_schema_cursor(cursor)

    def close(self) -> None:
        """Release resources owned by the catalog.

        Connections are operation-scoped, so the default implementation has
        no persistent resource to close. The method exists for API symmetry
        with SqliteDocumentCatalog.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- structured documents and chunks ---------------------------------

    def upsert_documents(self, documents: list[CatalogDocument]) -> None:
        """Insert or update documents by document ID."""
        if not documents:
            return
        rows = [
            (
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
            for document in documents
        ]
        statement = f"""
INSERT INTO {self._documents} (
    doc_id, source, title, doc_type, tenant_id, visibility,
    owner_group, status, created_at, updated_at, attributes_json
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
ON CONFLICT (doc_id) DO UPDATE SET
    source = EXCLUDED.source,
    title = EXCLUDED.title,
    doc_type = EXCLUDED.doc_type,
    tenant_id = EXCLUDED.tenant_id,
    visibility = EXCLUDED.visibility,
    owner_group = EXCLUDED.owner_group,
    status = EXCLUDED.status,
    created_at = EXCLUDED.created_at,
    updated_at = EXCLUDED.updated_at,
    attributes_json = EXCLUDED.attributes_json
""".strip()
        with self._database.cursor(write=True) as cursor:
            cursor.executemany(statement, rows)

    def upsert_chunks(self, chunks: list[CatalogChunk]) -> None:
        """Insert or update chunks; PostgreSQL refreshes search_vector."""
        if not chunks:
            return
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.chunk_index,
                chunk.section_path,
                chunk.text,
                chunk.content_hash,
                chunk.active,
            )
            for chunk in chunks
        ]
        statement = f"""
INSERT INTO {self._chunks} (
    chunk_id, doc_id, chunk_index, section_path, text, content_hash, active
) VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id,
    chunk_index = EXCLUDED.chunk_index,
    section_path = EXCLUDED.section_path,
    text = EXCLUDED.text,
    content_hash = EXCLUDED.content_hash,
    active = EXCLUDED.active
""".strip()
        with self._database.cursor(write=True) as cursor:
            cursor.executemany(statement, rows)

    def replace_chunks(self, doc_id: str, chunks: list[CatalogChunk]) -> list[str]:
        """Replace one document's chunks in one PostgreSQL transaction."""
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id must be a non-empty string")
        if any(chunk.doc_id != doc_id for chunk in chunks):
            raise ValueError("every replacement chunk must belong to doc_id")
        new_ids = {chunk.chunk_id for chunk in chunks}
        if len(new_ids) != len(chunks):
            raise ValueError("replacement chunk IDs must be unique")

        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.chunk_index,
                chunk.section_path,
                chunk.text,
                chunk.content_hash,
                chunk.active,
            )
            for chunk in chunks
        ]
        upsert = f"""
INSERT INTO {self._chunks} (
    chunk_id, doc_id, chunk_index, section_path, text, content_hash, active
) VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id,
    chunk_index = EXCLUDED.chunk_index,
    section_path = EXCLUDED.section_path,
    text = EXCLUDED.text,
    content_hash = EXCLUDED.content_hash,
    active = EXCLUDED.active
""".strip()
        with self._database.cursor(write=True) as cursor:
            cursor.execute(
                f"SELECT chunk_id FROM {self._chunks} WHERE doc_id = %s",
                (doc_id,),
            )
            old_ids = {str(_row_value(row, 0, "chunk_id")) for row in cursor.fetchall()}
            if rows:
                cursor.executemany(upsert, rows)
            removed = sorted(old_ids - new_ids)
            if removed:
                cursor.execute(
                    f"DELETE FROM {self._chunks} "
                    f"WHERE chunk_id IN ({_placeholders(len(removed))})",
                    tuple(removed),
                )
        return removed

    def delete_documents(self, doc_ids: list[str]) -> None:
        """Delete documents; foreign keys cascade chunks and ledger state."""
        unique_ids = list(dict.fromkeys(doc_ids))
        if not unique_ids:
            return
        placeholders = _placeholders(len(unique_ids))
        with self._database.cursor(write=True) as cursor:
            cursor.execute(
                f"DELETE FROM {self._documents} WHERE doc_id IN ({placeholders})",
                tuple(unique_ids),
            )

    def find(
        self,
        filter: MetadataFilter | None = None,
        scope: RetrievalScope | None = None,
        limit: int = 100,
    ) -> list[CatalogDocument]:
        """Find documents with metadata filters and scope pushed into SQL."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        clauses, parameters = _scope_clauses(scope or _UNRESTRICTED)
        if filter:
            filter_clauses, filter_parameters = _filter_clauses(filter)
            clauses.extend(filter_clauses)
            parameters.extend(filter_parameters)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        fields = ", ".join(f"d.{field}" for field in _DOCUMENT_FIELDS)
        statement = (
            f"SELECT {fields} FROM {self._documents} AS d "
            f"{where} ORDER BY d.doc_id LIMIT %s"
        )
        with self._database.cursor() as cursor:
            cursor.execute(statement, (*parameters, limit))
            rows = cursor.fetchall()
        return [_document_from_row(row) for row in rows]

    def get_chunks(self, chunk_ids: list[str]) -> list[CatalogChunk]:
        """Return known chunks in requested-ID order."""
        if not chunk_ids:
            return []
        unique_ids = list(dict.fromkeys(chunk_ids))
        placeholders = _placeholders(len(unique_ids))
        fields = ", ".join(_CHUNK_FIELDS)
        with self._database.cursor() as cursor:
            cursor.execute(
                f"SELECT {fields} FROM {self._chunks} "
                f"WHERE chunk_id IN ({placeholders})",
                tuple(unique_ids),
            )
            rows = cursor.fetchall()
        by_id = {
            str(_row_value(row, 0, "chunk_id")): _chunk_from_row(row) for row in rows
        }
        return [by_id[id_] for id_ in chunk_ids if id_ in by_id]

    # -- lexical retrieval -------------------------------------------------

    def search_lexical(
        self,
        query: str,
        k: int = 10,
        filter: MetadataFilter | None = None,
        scope: RetrievalScope | None = None,
    ) -> list[RankedHit]:
        """Rank active chunks using PostgreSQL tsvector full-text search."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("lexical query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        clauses, parameters = _scope_clauses(scope or _UNRESTRICTED)
        if filter:
            filter_clauses, filter_parameters = _filter_clauses(filter)
            clauses.extend(filter_clauses)
            parameters.extend(filter_parameters)
        extra = f" AND {' AND '.join(clauses)}" if clauses else ""
        statement = f"""
WITH search_query AS (
    SELECT websearch_to_tsquery(
        {self._text_search_config_sql}::regconfig, %s
    ) AS value
)
SELECT
    c.chunk_id,
    ts_rank_cd(c.search_vector, search_query.value, 32)::double precision
        AS score
FROM {self._chunks} AS c
JOIN {self._documents} AS d ON d.doc_id = c.doc_id
CROSS JOIN search_query
WHERE c.search_vector @@ search_query.value
    AND c.active = TRUE{extra}
ORDER BY score DESC, c.chunk_id ASC
LIMIT %s
""".strip()
        try:
            with self._database.cursor() as cursor:
                cursor.execute(statement, (query, *parameters, k))
                rows = cursor.fetchall()
        except Exception as exc:
            if _is_lexical_unavailable(exc):
                raise LexicalUnavailableError(
                    f"PostgreSQL lexical search is unavailable: {exc}"
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
        fields = ", ".join(_EMBEDDING_FIELDS)
        statement = f"SELECT {fields} FROM {self._chunk_embeddings} WHERE space_id = %s"
        parameters: list[object] = [space_id]
        if chunk_ids is not None:
            if not chunk_ids:
                return {}
            unique_ids = list(dict.fromkeys(chunk_ids))
            statement += f" AND chunk_id IN ({_placeholders(len(unique_ids))})"
            parameters.extend(unique_ids)
        with self._database.cursor() as cursor:
            cursor.execute(statement, tuple(parameters))
            rows = cursor.fetchall()
        states = [_embedding_state_from_row(row) for row in rows]
        return {state.chunk_id: state for state in states}

    def mark_embedded(
        self, chunk_id: str, spec: EmbeddingSpec, content_hash: str
    ) -> None:
        """Record that a current vector exists for a chunk and space."""
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError("content_hash must be a non-empty string")
        statement = f"""
INSERT INTO {self._chunk_embeddings} (
    chunk_id, space_id, provider, model, dimension,
    embedding_version, content_hash, created_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id, space_id) DO UPDATE SET
    provider = EXCLUDED.provider,
    model = EXCLUDED.model,
    dimension = EXCLUDED.dimension,
    embedding_version = EXCLUDED.embedding_version,
    content_hash = EXCLUDED.content_hash,
    created_at = EXCLUDED.created_at
""".strip()
        with self._database.cursor(write=True) as cursor:
            cursor.execute(
                statement,
                (
                    chunk_id,
                    spec.space_id,
                    spec.provider,
                    spec.model,
                    spec.dimension,
                    spec.version,
                    content_hash,
                    self._now(),
                ),
            )

    def invalidate_embeddings(self, chunk_ids: list[str]) -> None:
        """Remove lifecycle state after authorization/filter metadata changes."""
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return
        with self._database.cursor(write=True) as cursor:
            cursor.execute(
                f"DELETE FROM {self._chunk_embeddings} "
                f"WHERE chunk_id IN ({_placeholders(len(unique_ids))})",
                tuple(unique_ids),
            )

    def stale_chunk_ids(self, spec: EmbeddingSpec) -> list[str]:
        """Return active chunks missing a current vector for the space."""
        statement = f"""
SELECT c.chunk_id
FROM {self._chunks} AS c
LEFT JOIN {self._chunk_embeddings} AS e
    ON e.chunk_id = c.chunk_id AND e.space_id = %s
WHERE c.active = TRUE
    AND (e.chunk_id IS NULL OR e.content_hash <> c.content_hash)
ORDER BY c.chunk_id
""".strip()
        with self._database.cursor() as cursor:
            cursor.execute(statement, (spec.space_id,))
            rows = cursor.fetchall()
        return [str(_row_value(row, 0, "chunk_id")) for row in rows]

    # -- durable budget facade ---------------------------------------------

    def reserve(
        self,
        charge: EmbeddingCharge,
        *,
        daily_limit_nanos: int | None,
        monthly_limit_nanos: int | None,
        ttl_seconds: float,
    ) -> BudgetReservationDecision:
        """Delegate atomic spend reservation to the budget component."""
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
        """Delegate reservation reconciliation to the budget component."""
        self._budget.commit(reservation, actual_charge)

    def release(self, reservation: BudgetReservation) -> None:
        """Delegate reservation release to the budget component."""
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
        """Delegate committed usage recording to the budget component."""
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
        cursor.execute(
            f"SELECT version, text_search_config FROM {self._catalog_schema} "
            "WHERE singleton = TRUE"
        )
        metadata_row = cursor.fetchone()
        if metadata_row is None:
            raise RuntimeError("PostgreSQL catalog schema metadata is missing")
        version = int(_row_value(metadata_row, 0, "version"))
        configured_search = str(_row_value(metadata_row, 1, "text_search_config"))
        if version != _SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported PostgreSQL catalog schema version {version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if configured_search != self.text_search_config:
            raise ValueError(
                "PostgreSQL catalog text-search configuration mismatch: "
                f"schema uses {configured_search!r}, catalog requested "
                f"{self.text_search_config!r}"
            )

        cursor.execute(
            """
SELECT attribute.attgenerated,
    pg_catalog.format_type(attribute.atttypid, attribute.atttypmod)
FROM pg_catalog.pg_attribute AS attribute
JOIN pg_catalog.pg_class AS relation
    ON relation.oid = attribute.attrelid
JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = relation.relnamespace
WHERE namespace.nspname = %s
    AND relation.relname = 'chunks'
    AND attribute.attname = 'search_vector'
    AND NOT attribute.attisdropped
""".strip(),
            (self.schema_name,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(
                f"PostgreSQL table {self._chunks} is missing its search_vector column"
            )
        generated = str(_row_value(row, 0, "attgenerated"))
        data_type = str(_row_value(row, 1, "format_type"))
        if generated != "s" or data_type != "tsvector":
            raise ValueError(
                f"PostgreSQL table {self._chunks} must use a stored generated "
                "tsvector search_vector column"
            )

        cursor.execute(
            """
SELECT access_method.amname, pg_catalog.pg_get_indexdef(index_relation.oid)
FROM pg_catalog.pg_class AS index_relation
JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = index_relation.relnamespace
JOIN pg_catalog.pg_am AS access_method
    ON access_method.oid = index_relation.relam
WHERE namespace.nspname = %s
    AND index_relation.relname = 'idx_chunks_search_vector'
""".strip(),
            (self.schema_name,),
        )
        index_row = cursor.fetchone()
        if index_row is None:
            raise RuntimeError(
                f"PostgreSQL table {self._chunks} is missing its GIN lexical index"
            )
        access_method = str(_row_value(index_row, 0, "amname"))
        definition = str(_row_value(index_row, 1, "pg_get_indexdef"))
        if access_method != "gin" or "search_vector" not in definition:
            raise ValueError(
                f"PostgreSQL index {self.schema_name}.idx_chunks_search_vector "
                "must be a GIN index over search_vector"
            )


def _filter_clauses(
    filter: MetadataFilter,
) -> tuple[list[str], list[object]]:
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
    expression = f"d.{key}"
    if not isinstance(condition, dict):
        clauses.append(f"{expression} = %s")
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
        clauses.append(f"{expression} {sql_operator} %s")
        parameters.append(_text_filter_value(expected, f"{operator} filter"))


def _append_json_filter(
    clauses: list[str],
    parameters: list[object],
    key: str,
    condition: object,
) -> None:
    if not isinstance(condition, dict):
        clauses.append("d.attributes_json -> %s = %s::jsonb")
        parameters.extend((key, _json_filter_value(condition, "equality filter")))
        return
    if not condition:
        raise ValueError(f"metadata filter for {key!r} cannot be empty")
    for operator, expected in condition.items():
        if operator == "$in":
            candidates = _membership_values(expected)
            serialized = [
                _json_filter_value(candidate, "$in filter") for candidate in candidates
            ]
            placeholders = ", ".join("%s::jsonb" for _ in serialized)
            clauses.append(f"d.attributes_json -> %s IN ({placeholders})")
            parameters.extend((key, *serialized))
            continue
        sql_operator = _ORDERED_OPERATORS.get(operator)
        if sql_operator is None:
            raise ValueError(f"unsupported metadata filter operator: {operator!r}")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or not math.isfinite(expected)
        ):
            raise ValueError(f"{operator} requires a finite numeric value")
        clauses.append(
            "CASE WHEN jsonb_typeof(d.attributes_json -> %s) = 'number' "
            "THEN (d.attributes_json ->> %s)::numeric END "
            f"{sql_operator} %s"
        )
        parameters.extend((key, key, expected))


def _scope_clauses(scope: RetrievalScope) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    parameters: list[object] = []
    if scope.tenant_id is not None:
        clauses.append("(d.tenant_id IS NULL OR d.tenant_id = %s)")
        parameters.append(scope.tenant_id)
    if scope.visibility is not None:
        clauses.append(
            "(d.visibility IS NULL OR "
            f"d.visibility IN ({_placeholders(len(scope.visibility))}))"
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


def _json_filter_value(value: object, label: str) -> str:
    if isinstance(value, bool) or isinstance(value, (str, int)):
        pass
    elif isinstance(value, float) and math.isfinite(value):
        pass
    else:
        raise ValueError(f"{label} requires a finite scalar metadata value")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _serialize_attributes(attributes: Mapping[str, MetadataValue]) -> str:
    for key, value in attributes.items():
        if not isinstance(key, str):
            raise ValueError("attribute keys must be strings")
        _json_filter_value(value, "document attribute")
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
        active=bool(_row_value(row, 6, "active")),
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


def _is_lexical_unavailable(exc: Exception) -> bool:
    sqlstate = getattr(exc, "sqlstate", None)
    if sqlstate in _LEXICAL_UNAVAILABLE_SQLSTATES:
        return True
    diagnostic = getattr(exc, "diag", None)
    return getattr(diagnostic, "sqlstate", None) in _LEXICAL_UNAVAILABLE_SQLSTATES


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{label} must start with a letter or underscore and contain only "
            "letters, numbers, and underscores"
        )
    if len(value.encode("utf-8")) > 63:
        raise ValueError(f"{label} cannot exceed 63 UTF-8 bytes")


def _validate_text_search_config(value: str) -> None:
    if not isinstance(value, str) or not _TEXT_SEARCH_CONFIG.fullmatch(value):
        raise ValueError(
            "text_search_config must be an identifier or schema-qualified identifier"
        )
    if any(len(part.encode("utf-8")) > 63 for part in value.split(".")):
        raise ValueError("text_search_config identifiers cannot exceed 63 UTF-8 bytes")


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier}"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _placeholders(count: int) -> str:
    return ", ".join("%s" for _ in range(count))
