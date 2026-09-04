"""Lifecycle-aware document ingestion orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from vectorstore.catalog.base import CatalogChunk, CatalogDocument, DocumentCatalog
from vectorstore.embeddings.base import EmbeddingProvider, EmbeddingSpec
from vectorstore.embeddings.policy import (
    EmbeddingRouter,
    NoProviderAvailableError,
    ProviderSelection,
)
from vectorstore.models import Chunk, MetadataValue
from vectorstore.records import Record, semantic_projection
from vectorstore.stores.base import VectorStore

from .base import Source, SourceAdapter
from .chunkers import Chunker, WholeRecordChunker

type Projector = Callable[[Record], str]

_DOCUMENT_FIELDS = (
    "source",
    "title",
    "doc_type",
    "tenant_id",
    "visibility",
    "owner_group",
    "status",
    "created_at",
    "updated_at",
)
_DOCUMENT_FIELD_SET = frozenset(("doc_id", *_DOCUMENT_FIELDS))


class IngestionError(RuntimeError):
    """Ingestion could not complete an embedding or storage operation."""


class PrimaryProviderRequiredError(IngestionError):
    """Policy rejected the primary provider while primary-only ingest was set."""


class FallbackIndexMode(StrEnum):
    """When the fallback provider's independent vector space is populated."""

    EAGER = "eager"
    LAZY = "lazy"
    OFF = "off"


@dataclass(frozen=True)
class IngestionConfig:
    """Policy knobs for ingestion and embedding batches."""

    fallback_index: FallbackIndexMode | str = FallbackIndexMode.EAGER
    ingest_requires_primary: bool = True
    batch_size: int = 100

    def __post_init__(self) -> None:
        try:
            mode = FallbackIndexMode(self.fallback_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fallback_index must be 'eager', 'lazy', or 'off'"
            ) from exc
        object.__setattr__(self, "fallback_index", mode)
        if not isinstance(self.ingest_requires_primary, bool):
            raise ValueError("ingest_requires_primary must be a boolean")
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")


@dataclass(frozen=True)
class IngestionResult:
    """Counts and per-space lifecycle outcomes from one ingestion run."""

    document_count: int
    chunk_count: int
    removed_chunk_count: int = 0
    embedded_by_space: Mapping[str, int] = field(default_factory=dict)
    skipped_by_space: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label in ("document_count", "chunk_count", "removed_chunk_count"):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        object.__setattr__(
            self,
            "embedded_by_space",
            MappingProxyType(_validated_counts(self.embedded_by_space)),
        )
        object.__setattr__(
            self,
            "skipped_by_space",
            MappingProxyType(_validated_counts(self.skipped_by_space)),
        )

    @property
    def embedded_count(self) -> int:
        """Total vector writes across all embedding spaces."""
        return sum(self.embedded_by_space.values())

    @property
    def skipped_embedding_count(self) -> int:
        """Total current vectors skipped across targeted spaces."""
        return sum(self.skipped_by_space.values())


@dataclass(frozen=True)
class ReembeddingResult:
    """Outcome of repairing all stale vectors in one embedding space."""

    space_id: str
    stale_count: int
    embedded_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.space_id, str) or not self.space_id:
            raise ValueError("space_id must be a non-empty string")
        for label in ("stale_count", "embedded_count"):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")


@dataclass(frozen=True)
class _PreparedRecord:
    record: Record
    document: CatalogDocument
    chunks: tuple[CatalogChunk, ...]


class IngestionPipeline:
    """Project, chunk, catalog, and incrementally embed source records.

    The catalog is written before dense indexing. A failed vector operation
    therefore leaves the affected lifecycle row missing or stale, and a later
    :meth:`reembed_stale` call can safely repair it. Each configured provider
    is paired with exactly one store through its ``EmbeddingSpec.space_id``.
    """

    def __init__(
        self,
        catalog: DocumentCatalog,
        stores: Mapping[str, VectorStore] | None = None,
        router: EmbeddingRouter | None = None,
        *,
        chunker: Chunker | None = None,
        project: Projector = semantic_projection,
        config: IngestionConfig | None = None,
    ) -> None:
        if not callable(project):
            raise TypeError("project must be callable")
        self._catalog = catalog
        self._stores = dict(stores or {})
        self._router = router
        self._chunker = chunker or WholeRecordChunker()
        self._project = project
        self._config = config or IngestionConfig()
        self._providers = self._provider_map(router)
        self._validate_spaces()

    @property
    def config(self) -> IngestionConfig:
        """The immutable ingestion configuration."""
        return self._config

    def ingest_source(self, adapter: SourceAdapter, source: Source) -> IngestionResult:
        """Adapt and ingest one external file or directory source."""
        return self.ingest(adapter.iter_records(source))

    def ingest(self, records: Iterable[Record]) -> IngestionResult:
        """Ingest records and write only missing or stale vectors."""
        prepared = self._prepare(records)
        documents = [item.document for item in prepared]
        all_chunks = [chunk for item in prepared for chunk in item.chunks]

        changed_metadata_ids = self._metadata_changed_chunk_ids(documents, all_chunks)
        if changed_metadata_ids:
            # Dense filters carry a snapshot of document metadata. Remove that
            # snapshot before changing authorization-relevant catalog fields;
            # missing lifecycle rows make the subsequent write mandatory.
            self._catalog.invalidate_embeddings(changed_metadata_ids)
            for space_id, store in self._stores.items():
                try:
                    store.delete(changed_metadata_ids)
                except Exception as exc:
                    raise IngestionError(
                        f"could not invalidate dense metadata in {space_id!r}: {exc}"
                    ) from exc

        if documents:
            self._catalog.upsert_documents(documents)

        removed_ids: list[str] = []
        for item in prepared:
            removed_ids.extend(
                self._catalog.replace_chunks(item.document.doc_id, list(item.chunks))
            )
        if removed_ids:
            unique_removed = list(dict.fromkeys(removed_ids))
            for space_id, store in self._stores.items():
                try:
                    store.delete(unique_removed)
                except Exception as exc:
                    raise IngestionError(
                        f"could not prune superseded vectors from {space_id!r}: {exc}"
                    ) from exc

        stats_embedded: dict[str, int] = {}
        stats_skipped: dict[str, int] = {}
        active_chunks = [chunk for chunk in all_chunks if chunk.active]
        documents_by_id = {document.doc_id: document for document in documents}
        initial_stale: dict[str, list[CatalogChunk]] = {}

        for provider in self._target_providers():
            stale = self._stale_chunks(provider.spec, active_chunks)
            initial_stale[provider.spec.space_id] = stale
            stats_skipped[provider.spec.space_id] = len(active_chunks) - len(stale)

        primary = self._router.primary if self._router is not None else None
        if primary is not None and primary.spec.space_id in self._stores:
            stale_primary = initial_stale.get(primary.spec.space_id)
            if stale_primary is None:
                stale_primary = self._stale_chunks(primary.spec, active_chunks)
                initial_stale[primary.spec.space_id] = stale_primary
                stats_skipped[primary.spec.space_id] = len(active_chunks) - len(
                    stale_primary
                )
            self._embed_primary_batches(
                stale_primary,
                documents_by_id,
                stats_embedded,
                stats_skipped,
            )

        fallback = self._router.fallback if self._router is not None else None
        if (
            fallback is not None
            and self.config.fallback_index is FallbackIndexMode.EAGER
            and fallback.spec.space_id in self._stores
        ):
            remaining = self._stale_chunks(fallback.spec, active_chunks)
            self._embed_direct_batches(
                fallback,
                remaining,
                documents_by_id,
                stats_embedded,
            )

        return IngestionResult(
            document_count=len(prepared),
            chunk_count=len(all_chunks),
            removed_chunk_count=len(set(removed_ids)),
            embedded_by_space=stats_embedded,
            skipped_by_space=stats_skipped,
        )

    def reembed_stale(self, space: str | EmbeddingSpec) -> ReembeddingResult:
        """Repair every active missing/stale vector in one configured space."""
        space_id = space.space_id if isinstance(space, EmbeddingSpec) else space
        if not isinstance(space_id, str) or not space_id:
            raise ValueError("space must be an EmbeddingSpec or non-empty space ID")
        provider = self._providers.get(space_id)
        store = self._stores.get(space_id)
        if provider is None or store is None:
            raise ValueError(f"embedding space {space_id!r} is not configured")

        stale_ids = self._catalog.stale_chunk_ids(provider.spec)
        chunks = self._catalog.get_chunks(stale_ids)
        documents = self._documents_for_chunks(chunks)

        if self._router is not None and provider is self._router.primary:
            embedded = self._reembed_primary(chunks, documents)
        else:
            embedded = self._embed_direct_batches(provider, chunks, documents, {})
        return ReembeddingResult(
            space_id=space_id,
            stale_count=len(stale_ids),
            embedded_count=embedded,
        )

    def _prepare(self, records: Iterable[Record]) -> list[_PreparedRecord]:
        materialized = list(records)
        seen_records: set[str] = set()
        seen_chunks: set[str] = set()
        prepared: list[_PreparedRecord] = []
        for record in materialized:
            if not isinstance(record, Record):
                raise TypeError("ingest records must be Record instances")
            if record.id in seen_records:
                raise ValueError(
                    f"duplicate record ID in ingestion batch: {record.id!r}"
                )
            seen_records.add(record.id)

            projection = self._project(record)
            if not isinstance(projection, str) or not projection.strip():
                raise ValueError(
                    f"projector returned no semantic text for record {record.id!r}"
                )
            chunks = self._chunker.chunk(record, projection)
            for chunk in chunks:
                if chunk.doc_id != record.id:
                    raise ValueError(
                        f"chunk {chunk.chunk_id!r} does not belong to record "
                        f"{record.id!r}"
                    )
                if not chunk.active:
                    raise ValueError("ingestion chunkers must emit active chunks")
                if chunk.chunk_id in seen_chunks:
                    raise ValueError(
                        f"duplicate chunk ID in ingestion batch: {chunk.chunk_id!r}"
                    )
                seen_chunks.add(chunk.chunk_id)
            prepared.append(
                _PreparedRecord(
                    record=record,
                    document=_catalog_document(record),
                    chunks=tuple(chunks),
                )
            )
        return prepared

    def _target_providers(self) -> list[EmbeddingProvider]:
        if self._router is None:
            return []
        providers: list[EmbeddingProvider] = []
        primary = self._router.primary
        if primary.spec.space_id in self._stores:
            providers.append(primary)
        fallback = self._router.fallback
        if (
            fallback is not None
            and fallback.spec.space_id in self._stores
            and self.config.fallback_index is FallbackIndexMode.EAGER
        ):
            providers.append(fallback)
        return providers

    def _metadata_changed_chunk_ids(
        self,
        documents: Sequence[CatalogDocument],
        chunks: Sequence[CatalogChunk],
    ) -> list[str]:
        if not documents:
            return []
        doc_ids = [document.doc_id for document in documents]
        existing = self._catalog.find({"doc_id": {"$in": doc_ids}}, limit=len(doc_ids))
        previous_by_id = {document.doc_id: document for document in existing}
        changed_docs = {
            document.doc_id
            for document in documents
            if (previous := previous_by_id.get(document.doc_id)) is not None
            and previous != document
        }
        return [chunk.chunk_id for chunk in chunks if chunk.doc_id in changed_docs]

    def _embed_primary_batches(
        self,
        chunks: list[CatalogChunk],
        documents: Mapping[str, CatalogDocument],
        embedded: dict[str, int],
        skipped: dict[str, int],
    ) -> int:
        assert self._router is not None
        total = 0
        for batch in _batches(chunks, self.config.batch_size):
            texts = [chunk.text for chunk in batch]
            try:
                selection = self._router.select("ingest", texts=texts)
            except NoProviderAvailableError as exc:
                raise PrimaryProviderRequiredError(str(exc)) from exc

            if selection.provider is self._router.primary:
                try:
                    count = self._embed_selected_primary(selection, batch, documents)
                except Exception as exc:
                    if self.config.ingest_requires_primary:
                        if isinstance(exc, IngestionError):
                            raise
                        raise IngestionError(str(exc)) from exc
                    fallback = self._router.fallback
                    if (
                        fallback is None
                        or self.config.fallback_index is FallbackIndexMode.OFF
                        or fallback.spec.space_id not in self._stores
                    ):
                        if isinstance(exc, IngestionError):
                            raise
                        raise IngestionError(str(exc)) from exc
                    self._ensure_skip_stat(fallback, batch, skipped)
                    count = self._embed_missing_direct(
                        fallback, batch, documents, embedded
                    )
                    total += count
                    continue
                embedded[selection.spec.space_id] = (
                    embedded.get(selection.spec.space_id, 0) + count
                )
                total += count
                continue

            if self.config.ingest_requires_primary:
                self._router.release_reservation(selection.reservation)
                raise PrimaryProviderRequiredError(
                    "primary provider is required for ingestion but routing "
                    f"selected {selection.spec.space_id!r} "
                    f"({selection.reason})"
                )
            if self.config.fallback_index is FallbackIndexMode.OFF:
                raise IngestionError(
                    "routing selected the fallback provider but fallback indexing "
                    "is off"
                )
            if selection.spec.space_id not in self._stores:
                raise IngestionError(
                    f"no vector store is configured for routed fallback space "
                    f"{selection.spec.space_id!r}"
                )
            self._ensure_skip_stat(selection.provider, batch, skipped)
            count = self._embed_missing_direct(
                selection.provider, batch, documents, embedded
            )
            total += count
        return total

    def _embed_selected_primary(
        self,
        selection: ProviderSelection,
        chunks: list[CatalogChunk],
        documents: Mapping[str, CatalogDocument],
    ) -> int:
        assert self._router is not None
        try:
            result = selection.provider.embed_texts_with_usage(
                [chunk.text for chunk in chunks]
            )
        except Exception as exc:
            self._router.record_failure(selection.reservation)
            raise IngestionError(
                f"primary embedding failed ({selection.spec.space_id}): {exc}"
            ) from exc

        tokens = (
            result.usage.input_tokens
            if result.usage is not None
            else selection.provider.estimate_tokens([chunk.text for chunk in chunks])
        )
        self._router.record_usage(tokens, reservation=selection.reservation)
        vectors = result.vectors
        if len(vectors) != len(chunks):
            raise IngestionError(
                "embedding provider returned a different number of vectors than texts"
            )
        self._write_vectors(selection.provider, chunks, vectors, documents)
        return len(chunks)

    def _embed_missing_direct(
        self,
        provider: EmbeddingProvider,
        chunks: list[CatalogChunk],
        documents: Mapping[str, CatalogDocument],
        embedded: dict[str, int],
    ) -> int:
        missing = self._stale_chunks(provider.spec, chunks)
        count = self._embed_direct_batches(provider, missing, documents, embedded)
        return count

    def _embed_direct_batches(
        self,
        provider: EmbeddingProvider,
        chunks: list[CatalogChunk],
        documents: Mapping[str, CatalogDocument],
        embedded: dict[str, int],
    ) -> int:
        total = 0
        for batch in _batches(chunks, self.config.batch_size):
            try:
                result = provider.embed_texts_with_usage(
                    [chunk.text for chunk in batch]
                )
            except Exception as exc:
                raise IngestionError(
                    f"embedding failed ({provider.spec.space_id}): {exc}"
                ) from exc
            vectors = result.vectors
            if len(vectors) != len(batch):
                raise IngestionError(
                    "embedding provider returned a different number of vectors than "
                    "texts"
                )
            self._write_vectors(provider, batch, vectors, documents)
            total += len(batch)
            embedded[provider.spec.space_id] = embedded.get(
                provider.spec.space_id, 0
            ) + len(batch)
        return total

    def _write_vectors(
        self,
        provider: EmbeddingProvider,
        chunks: list[CatalogChunk],
        vectors: list[list[float]],
        documents: Mapping[str, CatalogDocument],
    ) -> None:
        store = self._stores[provider.spec.space_id]
        dense_chunks: list[Chunk] = []
        for chunk in chunks:
            document = documents.get(chunk.doc_id)
            if document is None:
                raise IngestionError(
                    f"catalog document {chunk.doc_id!r} is unavailable for "
                    "dense metadata"
                )
            dense_chunks.append(_vector_chunk(chunk, document))
        try:
            store.upsert(dense_chunks, vectors)
        except Exception as exc:
            raise IngestionError(
                f"vector upsert failed ({provider.spec.space_id}): {exc}"
            ) from exc
        for chunk in chunks:
            assert chunk.content_hash is not None
            self._catalog.mark_embedded(
                chunk.chunk_id, provider.spec, chunk.content_hash
            )

    def _reembed_primary(
        self,
        chunks: list[CatalogChunk],
        documents: Mapping[str, CatalogDocument],
    ) -> int:
        assert self._router is not None
        total = 0
        for batch in _batches(chunks, self.config.batch_size):
            try:
                selection = self._router.select(
                    "ingest", texts=[chunk.text for chunk in batch]
                )
            except NoProviderAvailableError as exc:
                raise IngestionError(str(exc)) from exc
            if selection.provider is not self._router.primary:
                raise IngestionError(
                    f"cannot repair primary space: routing selected "
                    f"{selection.spec.space_id!r} ({selection.reason})"
                )
            total += self._embed_selected_primary(selection, batch, documents)
        return total

    def _stale_chunks(
        self, spec: EmbeddingSpec, chunks: Sequence[CatalogChunk]
    ) -> list[CatalogChunk]:
        ids = [chunk.chunk_id for chunk in chunks]
        states = self._catalog.embedding_state(spec.space_id, ids)
        return [
            chunk
            for chunk in chunks
            if (state := states.get(chunk.chunk_id)) is None
            or state.content_hash != chunk.content_hash
            or state.provider != spec.provider
            or state.model != spec.model
            or state.dimension != spec.dimension
            or state.version != spec.version
        ]

    def _documents_for_chunks(
        self, chunks: Sequence[CatalogChunk]
    ) -> dict[str, CatalogDocument]:
        doc_ids = list(dict.fromkeys(chunk.doc_id for chunk in chunks))
        if not doc_ids:
            return {}
        documents = self._catalog.find({"doc_id": {"$in": doc_ids}}, limit=len(doc_ids))
        by_id = {document.doc_id: document for document in documents}
        missing = [doc_id for doc_id in doc_ids if doc_id not in by_id]
        if missing:
            raise IngestionError(
                f"catalog documents unavailable for stale chunks: {missing!r}"
            )
        return by_id

    def _ensure_skip_stat(
        self,
        provider: EmbeddingProvider,
        chunks: Sequence[CatalogChunk],
        skipped: dict[str, int],
    ) -> None:
        if provider.spec.space_id in skipped:
            return
        stale = self._stale_chunks(provider.spec, chunks)
        skipped[provider.spec.space_id] = len(chunks) - len(stale)

    @staticmethod
    def _provider_map(
        router: EmbeddingRouter | None,
    ) -> dict[str, EmbeddingProvider]:
        if router is None:
            return {}
        providers = [router.primary]
        if router.fallback is not None:
            providers.append(router.fallback)
        return {provider.spec.space_id: provider for provider in providers}

    def _validate_spaces(self) -> None:
        if self._stores and self._router is None:
            raise ValueError("vector stores require an embedding router")
        unknown = set(self._stores) - set(self._providers)
        if unknown:
            raise ValueError(
                f"vector stores have no configured provider: {sorted(unknown)!r}"
            )
        for space_id, store in self._stores.items():
            spec = self._providers[space_id].spec
            if store.dimension is not None and store.dimension != spec.dimension:
                raise ValueError(
                    f"store registered for space {space_id!r} expects "
                    f"{store.dimension}-dimensional vectors but the provider "
                    f"produces {spec.dimension}"
                )


def _catalog_document(record: Record) -> CatalogDocument:
    structured = dict(record.structured)
    title = _structured_text(structured, "title") or _semantic_title(record)
    source = record.source or _structured_text(structured, "source")
    attributes = {
        key: value
        for key, value in structured.items()
        if key not in _DOCUMENT_FIELD_SET
    }
    return CatalogDocument(
        doc_id=record.id,
        source=source,
        title=title,
        doc_type=_structured_text(structured, "doc_type"),
        tenant_id=_structured_text(structured, "tenant_id"),
        visibility=_structured_text(structured, "visibility"),
        owner_group=_structured_text(structured, "owner_group"),
        status=_structured_text(structured, "status"),
        created_at=_structured_text(structured, "created_at"),
        updated_at=_structured_text(structured, "updated_at"),
        attributes=attributes,
    )


def _vector_chunk(chunk: CatalogChunk, document: CatalogDocument) -> Chunk:
    metadata: dict[str, MetadataValue] = dict(document.attributes)
    metadata["doc_id"] = document.doc_id
    for field_name in _DOCUMENT_FIELDS:
        value = getattr(document, field_name)
        if value is not None:
            metadata[field_name] = value
    metadata["chunk_index"] = chunk.chunk_index
    if chunk.section_path is not None:
        metadata["section_path"] = chunk.section_path
    return Chunk(id=chunk.chunk_id, text=chunk.text, metadata=metadata)


def _structured_text(structured: Mapping[str, MetadataValue], key: str) -> str | None:
    value = structured.get(key)
    return None if value is None else str(value)


def _semantic_title(record: Record) -> str | None:
    for label, value in record.semantic_fields.items():
        if label.strip().lower() == "title" and value.strip():
            return value.strip()
    return None


def _batches(
    chunks: Sequence[CatalogChunk], batch_size: int
) -> Iterable[list[CatalogChunk]]:
    for start in range(0, len(chunks), batch_size):
        yield list(chunks[start : start + batch_size])


def _validated_counts(counts: Mapping[str, int]) -> dict[str, int]:
    snapshot: dict[str, int] = {}
    for space_id, count in counts.items():
        if not isinstance(space_id, str) or not space_id:
            raise ValueError("embedding count keys must be non-empty space IDs")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("embedding counts must be non-negative integers")
        snapshot[space_id] = count
    return snapshot


__all__ = [
    "FallbackIndexMode",
    "IngestionConfig",
    "IngestionError",
    "IngestionPipeline",
    "IngestionResult",
    "PrimaryProviderRequiredError",
    "ReembeddingResult",
]
