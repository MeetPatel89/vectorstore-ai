"""SQLite-backed document catalog: structured find, FTS5 lexical, ledgers.

Uses only the standard library ``sqlite3`` module, so the default install
gets a complete catalog (structured + lexical + ledgers) with zero extra
dependencies. One database file per corpus; ``:memory:`` for tests.

Lexical search uses an FTS5 external-content virtual table over
``chunks.text`` kept in sync by triggers, so the full-text index is updated
in the same transaction as the row it indexes. Ranking uses ``bm25()``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from vectorstore.embeddings.base import EmbeddingSpec
from vectorstore.models import MetadataFilter

from .base import (
    CatalogChunk,
    CatalogDocument,
    EmbeddingState,
    LexicalUnavailableError,
    RankedHit,
    RetrievalScope,
)

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


_UNRESTRICTED = RetrievalScope()

# Document attributes with first-class columns; everything else lives in
# attributes_json and is filtered via json_extract.
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

_COMPARISON_SQL = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
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
    attributes_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_tenant_id ON documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    section_path TEXT,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT NOT NULL,
    space_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    embedding_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, space_id)
);

CREATE TABLE IF NOT EXISTS embedding_usage (
    date TEXT NOT NULL,
    provider TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    estimated_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embedding_usage_date ON embedding_usage(date);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS chunks_fts_after_insert
AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_after_delete
AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_after_update
AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""


def _fts_match_expression(query: str) -> str:
    """Convert a raw user query into a safe FTS5 MATCH expression.

    Every token is wrapped as an FTS5 string literal so user input can never
    produce a MATCH syntax error, and quoted phrases in the query are kept
    as phrase queries. Tokens are combined with FTS5's implicit AND.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("lexical query must be a non-empty string")

    tokens = re.findall(r'"[^"]*"|\S+', query)
    parts: list[str] = []
    for token in tokens:
        if token.startswith('"') and token.endswith('"') and len(token) >= 2:
            token = token[1:-1].strip()
            if not token:
                continue
        parts.append('"' + token.replace('"', '""') + '"')
    if not parts:
        raise ValueError("lexical query contains no searchable tokens")
    return " ".join(parts)


def _filter_clauses(
    filter: MetadataFilter,
) -> tuple[list[str], list[object]]:
    """Translate a MetadataFilter into parametrized SQL over documents ``d``.

    Known document attributes filter on their columns; any other key filters
    on ``attributes_json`` via ``json_extract`` with a parametrized path, so
    filter keys never reach the SQL text.
    """
    clauses: list[str] = []
    params: list[object] = []

    for key, condition in filter.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata filter keys must be non-empty strings")
        if key in _DOCUMENT_COLUMNS:
            expr = f"d.{key}"
        else:
            expr = "json_extract(d.attributes_json, ?)"
            params.append(f'$."{key}"')

        if not isinstance(condition, dict):
            clauses.append(f"{expr} = ?")
            params.append(_sql_value(condition))
            continue

        if not condition:
            raise ValueError(f"metadata filter for {key!r} cannot be empty")

        for operator, expected in condition.items():
            if operator == "$in":
                if not isinstance(expected, (list, tuple, set, frozenset)):
                    raise ValueError("$in requires a list-like value")
                values = [_sql_value(item) for item in expected]
                if not values:
                    raise ValueError("$in requires at least one value")
                placeholders = ", ".join("?" for _ in values)
                clauses.append(f"{expr} IN ({placeholders})")
                params.extend(values)
            elif operator in _COMPARISON_SQL:
                clauses.append(f"{expr} {_COMPARISON_SQL[operator]} ?")
                params.append(_sql_value(expected))
            else:
                raise ValueError(f"unsupported metadata filter operator: {operator!r}")

    return clauses, params


def _scope_clauses(scope: RetrievalScope) -> tuple[list[str], list[object]]:
    """Build SQL that enforces the authorization scope over documents ``d``.

    Documents without a tenant are shared across tenants; documents without
    a visibility label are visible to every scope.
    """
    clauses: list[str] = []
    params: list[object] = []
    if scope.tenant_id is not None:
        clauses.append("(d.tenant_id IS NULL OR d.tenant_id = ?)")
        params.append(scope.tenant_id)
    if scope.visibility is not None:
        placeholders = ", ".join("?" for _ in scope.visibility)
        clauses.append(f"(d.visibility IS NULL OR d.visibility IN ({placeholders}))")
        params.extend(scope.visibility)
    return clauses, params


def _sql_value(value: object) -> object:
    # SQLite stores booleans as integers; convert so equality works both for
    # column values and for json_extract results.
    if isinstance(value, bool):
        return int(value)
    return value


class SqliteDocumentCatalog:
    """A :class:`~vectorstore.catalog.base.DocumentCatalog` backed by SQLite.

    Also satisfies the :class:`~vectorstore.embeddings.policy.BudgetLedger`
    protocol with durable aggregates, so it can be passed directly to an
    :class:`~vectorstore.embeddings.policy.EmbeddingRouter` as its ledger.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        now: Clock = _utc_now,
    ) -> None:
        self._now = now
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path = str(path)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        with self._connection:
            self._connection.executescript(_SCHEMA)
            self._fts_available = self._try_create_fts()

    def _try_create_fts(self) -> bool:
        try:
            self._connection.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower():
                return False
            raise
        return True

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- structured documents and chunks ------------------------------------

    def upsert_documents(self, documents: list[CatalogDocument]) -> None:
        """Insert new documents and replace existing documents by ID."""
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
                json.dumps(document.attributes, sort_keys=True),
            )
            for document in documents
        ]
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO documents (
                    doc_id, source, title, doc_type, tenant_id, visibility,
                    owner_group, status, created_at, updated_at, attributes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (doc_id) DO UPDATE SET
                    source = excluded.source,
                    title = excluded.title,
                    doc_type = excluded.doc_type,
                    tenant_id = excluded.tenant_id,
                    visibility = excluded.visibility,
                    owner_group = excluded.owner_group,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    attributes_json = excluded.attributes_json
                """,
                rows,
            )

    def upsert_chunks(self, chunks: list[CatalogChunk]) -> None:
        """Insert new chunks and replace existing chunks by ID."""
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.chunk_index,
                chunk.section_path,
                chunk.text,
                chunk.content_hash,
                int(chunk.active),
            )
            for chunk in chunks
        ]
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, chunk_index, section_path,
                    text, content_hash, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    chunk_index = excluded.chunk_index,
                    section_path = excluded.section_path,
                    text = excluded.text,
                    content_hash = excluded.content_hash,
                    active = excluded.active
                """,
                rows,
            )

    def delete_documents(self, doc_ids: list[str]) -> None:
        """Delete documents and all associated chunk and ledger rows."""
        if not doc_ids:
            return
        placeholders = ", ".join("?" for _ in doc_ids)
        with self._connection:
            self._connection.execute(
                f"""
                DELETE FROM chunk_embeddings WHERE chunk_id IN (
                    SELECT chunk_id FROM chunks WHERE doc_id IN ({placeholders})
                )
                """,
                doc_ids,
            )
            # Chunk rows cascade; the FTS delete trigger keeps the index in sync.
            self._connection.execute(
                f"DELETE FROM documents WHERE doc_id IN ({placeholders})",
                doc_ids,
            )

    def find(
        self,
        filter: MetadataFilter | None = None,
        scope: RetrievalScope | None = None,
        limit: int = 100,
    ) -> list[CatalogDocument]:
        """Find documents matching structured filters and authorization scope."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        clauses, params = _scope_clauses(scope or _UNRESTRICTED)
        if filter:
            filter_clauses, filter_params = _filter_clauses(filter)
            clauses.extend(filter_clauses)
            params.extend(filter_params)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            f"SELECT * FROM documents AS d {where} ORDER BY d.doc_id LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_chunks(self, chunk_ids: list[str]) -> list[CatalogChunk]:
        """Return known chunks in requested-ID order."""
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        rows = self._connection.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        by_id = {row["chunk_id"]: self._row_to_chunk(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    # -- lexical retrieval ---------------------------------------------------

    def search_lexical(
        self,
        query: str,
        k: int = 10,
        filter: MetadataFilter | None = None,
        scope: RetrievalScope | None = None,
    ) -> list[RankedHit]:
        """Rank active chunks with SQLite FTS5 lexical search."""
        if not self._fts_available:
            raise LexicalUnavailableError(
                "this SQLite build does not include the FTS5 extension"
            )
        if k <= 0:
            raise ValueError("k must be greater than zero")

        match = _fts_match_expression(query)
        clauses, params = _scope_clauses(scope or _UNRESTRICTED)
        if filter:
            filter_clauses, filter_params = _filter_clauses(filter)
            clauses.extend(filter_clauses)
            params.extend(filter_params)
        extra = f"AND {' AND '.join(clauses)}" if clauses else ""

        try:
            rows = self._connection.execute(
                f"""
                SELECT c.chunk_id AS chunk_id, bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                JOIN chunks AS c ON c.rowid = chunks_fts.rowid
                JOIN documents AS d ON d.doc_id = c.doc_id
                WHERE chunks_fts MATCH ? AND c.active = 1 {extra}
                ORDER BY bm25_score
                LIMIT ?
                """,
                [match, *params, k],
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise LexicalUnavailableError(f"lexical search failed: {exc}") from exc

        # bm25() returns lower-is-better values; negate so higher is better.
        return [
            RankedHit(
                chunk_id=row["chunk_id"],
                rank=position,
                score=-row["bm25_score"],
            )
            for position, row in enumerate(rows, start=1)
        ]

    # -- embedding lifecycle ledger -------------------------------------------

    def embedding_state(
        self, space_id: str, chunk_ids: list[str] | None = None
    ) -> dict[str, EmbeddingState]:
        """Return recorded embedding state for an embedding space."""
        sql = "SELECT * FROM chunk_embeddings WHERE space_id = ?"
        params: list[object] = [space_id]
        if chunk_ids is not None:
            if not chunk_ids:
                return {}
            placeholders = ", ".join("?" for _ in chunk_ids)
            sql += f" AND chunk_id IN ({placeholders})"
            params.extend(chunk_ids)
        rows = self._connection.execute(sql, params).fetchall()
        return {
            row["chunk_id"]: EmbeddingState(
                chunk_id=row["chunk_id"],
                space_id=row["space_id"],
                provider=row["provider"],
                model=row["model"],
                dimension=row["dimension"],
                version=row["embedding_version"],
                content_hash=row["content_hash"],
                created_at=row["created_at"],
            )
            for row in rows
        }

    def mark_embedded(
        self, chunk_id: str, spec: EmbeddingSpec, content_hash: str
    ) -> None:
        """Record a chunk as embedded in the supplied embedding space."""
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError("content_hash must be a non-empty string")
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO chunk_embeddings (
                    chunk_id, space_id, provider, model, dimension,
                    embedding_version, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (chunk_id, space_id) DO UPDATE SET
                    provider = excluded.provider,
                    model = excluded.model,
                    dimension = excluded.dimension,
                    embedding_version = excluded.embedding_version,
                    content_hash = excluded.content_hash,
                    created_at = excluded.created_at
                """,
                (
                    chunk_id,
                    spec.space_id,
                    spec.provider,
                    spec.model,
                    spec.dimension,
                    spec.version,
                    content_hash,
                    self._now().isoformat(),
                ),
            )

    def stale_chunk_ids(self, spec: EmbeddingSpec) -> list[str]:
        """Return active chunks missing a current embedding for *spec*."""
        rows = self._connection.execute(
            """
            SELECT c.chunk_id AS chunk_id
            FROM chunks AS c
            LEFT JOIN chunk_embeddings AS e
                ON e.chunk_id = c.chunk_id AND e.space_id = ?
            WHERE c.active = 1
              AND (e.chunk_id IS NULL OR e.content_hash != c.content_hash)
            ORDER BY c.chunk_id
            """,
            (spec.space_id,),
        ).fetchall()
        return [row["chunk_id"] for row in rows]

    # -- budget ledger (BudgetLedger protocol) ---------------------------------

    def record(self, provider: str, tokens: int, usd: float) -> None:
        """Record token usage and estimated cost for the current UTC day."""
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        if usd < 0:
            raise ValueError("usd must not be negative")
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO embedding_usage (date, provider, tokens, estimated_usd)
                VALUES (?, ?, ?, ?)
                """,
                (self._now().strftime("%Y-%m-%d"), provider, tokens, usd),
            )

    def spent_today(self) -> float:
        """Return total estimated embedding spend for the current UTC day."""
        row = self._connection.execute(
            "SELECT COALESCE(SUM(estimated_usd), 0.0) AS total "
            "FROM embedding_usage WHERE date = ?",
            (self._now().strftime("%Y-%m-%d"),),
        ).fetchone()
        return float(row["total"])

    def spent_month(self) -> float:
        """Return total estimated embedding spend for the current UTC month."""
        row = self._connection.execute(
            "SELECT COALESCE(SUM(estimated_usd), 0.0) AS total "
            "FROM embedding_usage WHERE date LIKE ?",
            (self._now().strftime("%Y-%m") + "-%",),
        ).fetchone()
        return float(row["total"])

    def tokens_today(self, provider: str) -> int:
        """Return tokens recorded today for *provider*."""
        row = self._connection.execute(
            "SELECT COALESCE(SUM(tokens), 0) AS total "
            "FROM embedding_usage WHERE date = ? AND provider = ?",
            (self._now().strftime("%Y-%m-%d"), provider),
        ).fetchone()
        return int(row["total"])

    # -- row mapping ------------------------------------------------------------

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> CatalogDocument:
        return CatalogDocument(
            doc_id=row["doc_id"],
            source=row["source"],
            title=row["title"],
            doc_type=row["doc_type"],
            tenant_id=row["tenant_id"],
            visibility=row["visibility"],
            owner_group=row["owner_group"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            attributes=json.loads(row["attributes_json"]),
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> CatalogChunk:
        return CatalogChunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            chunk_index=row["chunk_index"],
            section_path=row["section_path"],
            text=row["text"],
            content_hash=row["content_hash"],
            active=bool(row["active"]),
        )
