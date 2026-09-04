"""Document catalog contracts for structured, lexical, and lifecycle data.

A :class:`DocumentCatalog` is the system of record for searchable documents.
It owns three of the four retrieval concerns:

- structured source data (documents and chunks with filterable attributes),
- the lexical/full-text index (database-native SQLite FTS5, PostgreSQL
  tsvector/GIN, or Azure SQL Full-Text Search),
- the embedding lifecycle ledger (which vector was built from which content).

Budget accounting is a separate application boundary represented by
:class:`~vectorstore.embeddings.policy.BudgetLedger`. The bundled SQL catalog
facades also satisfy that protocol for backward compatibility and convenient
transactional persistence, but retrieval-only consumers do not depend on it.

Dense vectors themselves live in per-space
:class:`~vectorstore.stores.base.VectorStore` instances, never in the
catalog. Lexical search is part of this protocol rather than a separate
retriever hierarchy because a full-text index is a feature of the table the
documents live in; one protocol avoids two parallel abstractions over the
same storage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol, runtime_checkable

from vectorstore.embeddings.base import EmbeddingSpec
from vectorstore.models import MetadataFilter, MetadataValue, _freeze_metadata
from vectorstore.records import content_hash as _content_hash


class LexicalUnavailableError(RuntimeError):
    """The lexical/full-text index cannot serve queries.

    Raised so the retrieval orchestration layer can degrade to dense plus
    structured retrieval instead of failing the whole request.
    """


@dataclass(frozen=True)
class RetrievalScope:
    """Mandatory authorization boundary applied inside candidate generation.

    The scope is enforced in SQL, before or during candidate generation,
    never by post-filtering an unauthorized result set.

    - ``tenant_id``: when set, only documents belonging to this tenant or
      shared documents (``tenant_id`` unset) are visible.
    - ``visibility``: when set, only documents whose visibility label is in
      this collection, or documents with no visibility label, are visible.

    ``RetrievalScope()`` is the explicit unrestricted scope for
    single-tenant deployments.
    """

    tenant_id: str | None = None
    visibility: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.tenant_id is not None and (
            not isinstance(self.tenant_id, str) or not self.tenant_id
        ):
            raise ValueError("tenant_id must be a non-empty string or None")
        if self.visibility is not None:
            if not isinstance(self.visibility, tuple) or not self.visibility:
                raise ValueError("visibility must be a non-empty tuple or None")
            for label in self.visibility:
                if not isinstance(label, str) or not label:
                    raise ValueError("visibility labels must be non-empty strings")


@dataclass(frozen=True)
class CatalogDocument:
    """A source document's structured, filterable representation.

    First-class columns (``doc_type``, ``tenant_id``, ``status``, ...) cover
    the attributes retrieval filters on most often; anything else goes into
    ``attributes`` and remains filterable through the same metadata-filter
    syntax.
    """

    doc_id: str
    title: str | None = None
    source: str | None = None
    doc_type: str | None = None
    tenant_id: str | None = None
    visibility: str | None = None
    owner_group: str | None = None
    status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    attributes: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError("doc_id must be a non-empty string")
        for label in (
            "title",
            "source",
            "doc_type",
            "tenant_id",
            "visibility",
            "owner_group",
            "status",
            "created_at",
            "updated_at",
        ):
            value = getattr(self, label)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{label} must be a string or None")
        object.__setattr__(
            self,
            "attributes",
            _freeze_metadata(self.attributes, label="document attributes"),
        )


@dataclass(frozen=True)
class CatalogChunk:
    """One indexable unit of a document's semantic projection.

    ``content_hash`` identifies the exact text the chunk carries; the
    embedding ledger compares it against the hash a vector was built from to
    detect staleness. It is computed from ``text`` when not supplied.
    """

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int = 0
    section_path: str | None = None
    content_hash: str | None = None
    active: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError("doc_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("chunk text must be a non-empty string")
        if (
            not isinstance(self.chunk_index, int)
            or isinstance(self.chunk_index, bool)
            or self.chunk_index < 0
        ):
            raise ValueError("chunk_index must be a non-negative integer")
        if self.section_path is not None and not isinstance(self.section_path, str):
            raise ValueError("section_path must be a string or None")
        if self.content_hash is not None and (
            not isinstance(self.content_hash, str) or not self.content_hash
        ):
            raise ValueError("content_hash must be a non-empty string or None")
        if not isinstance(self.active, bool):
            raise ValueError("active must be a boolean")
        if self.content_hash is None:
            object.__setattr__(self, "content_hash", _content_hash(self.text))


@dataclass(frozen=True)
class RankedHit:
    """One lexical search hit: a chunk ID with its rank and score.

    ``rank`` is 1-based; ``score`` is normalized so higher is better
    (SQLite BM25 values are negated). Ranks are what fusion consumes; raw
    scores are kept for observability only and are not comparable across
    retrieval signals.
    """

    chunk_id: str
    rank: int
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or self.rank <= 0
        ):
            raise ValueError("rank must be a positive integer")
        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or not isfinite(self.score)
        ):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class EmbeddingState:
    """Ledger entry: which vector exists for one (chunk, embedding space).

    Answers the lifecycle questions directly: which provider/model/dimension
    generated the vector, when, and from which semantic content
    (``content_hash``). A vector is stale when its hash no longer matches
    the chunk's current hash.
    """

    chunk_id: str
    space_id: str
    provider: str
    model: str
    dimension: int
    version: str
    content_hash: str
    created_at: str


@runtime_checkable
class RetrievalCatalog(Protocol):
    """The read boundary required by hybrid retrieval.

    Keeping this protocol smaller than :class:`DocumentCatalog` lets query
    paths depend only on candidate generation and hydration, without also
    acquiring document-mutation or embedding-lifecycle responsibilities.
    """

    def find(
        self,
        filter: MetadataFilter | None = None,
        scope: RetrievalScope | None = None,
        limit: int = 100,
    ) -> list[CatalogDocument]:
        """Structured retrieval: documents matching the filter within scope."""

    def get_chunks(self, chunk_ids: list[str]) -> list[CatalogChunk]:
        """Hydrate chunks by ID, preserving the requested order."""

    def search_lexical(
        self,
        query: str,
        k: int = 10,
        filter: MetadataFilter | None = None,
        scope: RetrievalScope | None = None,
    ) -> list[RankedHit]:
        """Full-text search over chunk text, filtered and scoped in storage.

        Raises :class:`LexicalUnavailableError` when the full-text index
        cannot serve queries.
        """


@runtime_checkable
class DocumentCatalog(RetrievalCatalog, Protocol):
    """Mutable document catalog with embedding-lifecycle state.

    Implementations for SQLite, PostgreSQL, and Azure SQL push scope and
    metadata filters down into SQL. The bundled implementations separately
    satisfy :class:`~vectorstore.embeddings.policy.BudgetLedger`; budget methods
    deliberately do not belong to this document-storage contract.
    """

    # -- structured documents and chunks ---------------------------------

    def upsert_documents(self, documents: list[CatalogDocument]) -> None:
        """Insert or update documents by ``doc_id``."""

    def upsert_chunks(self, chunks: list[CatalogChunk]) -> None:
        """Insert or update chunks while keeping the lexical index in sync."""

    def replace_chunks(self, doc_id: str, chunks: list[CatalogChunk]) -> list[str]:
        """Replace one document's chunk set and return removed chunk IDs.

        Unchanged chunk IDs retain their embedding-ledger rows, which lets an
        ingestion pipeline skip current vectors. Rows no longer emitted by a
        chunker are deleted with their lifecycle state.
        """

    def delete_documents(self, doc_ids: list[str]) -> None:
        """Remove documents with their chunks, index entries, and ledger rows."""

    # -- embedding lifecycle ledger ---------------------------------------

    def embedding_state(
        self, space_id: str, chunk_ids: list[str] | None = None
    ) -> dict[str, EmbeddingState]:
        """Ledger entries for one embedding space, keyed by chunk ID."""

    def mark_embedded(
        self, chunk_id: str, spec: EmbeddingSpec, content_hash: str
    ) -> None:
        """Record that a vector for this chunk exists in ``spec``'s space."""

    def invalidate_embeddings(self, chunk_ids: list[str]) -> None:
        """Remove all lifecycle rows for chunks whose dense metadata changed."""

    def stale_chunk_ids(self, spec: EmbeddingSpec) -> list[str]:
        """Active chunks whose vector in ``spec``'s space is missing or stale."""
