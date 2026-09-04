from __future__ import annotations

from typing import Never, override

import pytest
from conftest import FakeEmbedding

from vectorstore import (
    CatalogChunk,
    EmbeddingRouter,
    FallbackIndexMode,
    IngestionConfig,
    IngestionError,
    IngestionPipeline,
    MarkdownSectionChunker,
    NumpyVectorStore,
    PrimaryProviderRequiredError,
    Record,
    RetrievalScope,
    SqliteDocumentCatalog,
    WordChunker,
    build_retriever,
)


class FailingEmbedding(FakeEmbedding):
    @override
    def embed_texts(self, texts: list[str]) -> Never:
        raise RuntimeError("offline")


def record(body: str = "Payment reports are missing.") -> Record:
    return Record(
        id="INC-1104",
        semantic_fields={"Title": "Missing reports", "Body": body},
        structured={
            "doc_type": "incident",
            "tenant_id": "acme",
            "visibility": "internal",
            "status": "open",
            "severity": 2,
        },
        source="incidents/1104.md",
    )


def pipeline(
    catalog: SqliteDocumentCatalog,
    primary: FakeEmbedding,
    fallback: FakeEmbedding | None = None,
    *,
    config: IngestionConfig | None = None,
    chunker: WordChunker | MarkdownSectionChunker | None = None,
) -> tuple[IngestionPipeline, NumpyVectorStore, NumpyVectorStore | None]:
    primary_store = NumpyVectorStore(dimension=primary.dimension)
    fallback_store = (
        NumpyVectorStore(dimension=fallback.dimension) if fallback is not None else None
    )
    stores = {primary.spec.space_id: primary_store}
    if fallback is not None and fallback_store is not None:
        stores[fallback.spec.space_id] = fallback_store
    return (
        IngestionPipeline(
            catalog,
            stores,
            EmbeddingRouter(primary, fallback),
            chunker=chunker,
            config=config,
        ),
        primary_store,
        fallback_store,
    )


def test_ingestion_writes_catalog_and_both_spaces_then_skips_current() -> None:
    primary = FakeEmbedding(dimension=24, provider="primary", model="p")
    fallback = FakeEmbedding(dimension=12, provider="fallback", model="f")
    with SqliteDocumentCatalog() as catalog:
        ingest, primary_store, fallback_store = pipeline(catalog, primary, fallback)

        first = ingest.ingest([record()])
        second = ingest.ingest([record()])

        assert first.document_count == 1
        assert first.chunk_count == 1
        assert first.embedded_by_space == {
            primary.spec.space_id: 1,
            fallback.spec.space_id: 1,
        }
        assert first.skipped_embedding_count == 0
        assert second.embedded_count == 0
        assert second.skipped_by_space == {
            primary.spec.space_id: 1,
            fallback.spec.space_id: 1,
        }
        assert primary_store.count() == 1
        assert fallback_store is not None and fallback_store.count() == 1
        (document,) = catalog.find({"doc_id": "INC-1104"})
        assert document.attributes == {"severity": 2}
        dense_chunk = primary_store.get(["INC-1104::chunk-0000"])[0]
        assert dense_chunk.metadata["tenant_id"] == "acme"
        assert dense_chunk.metadata["severity"] == 2


def test_changed_content_reembeds_only_stale_chunk() -> None:
    primary = FakeEmbedding(dimension=16)
    with SqliteDocumentCatalog() as catalog:
        ingest, _, _ = pipeline(catalog, primary)
        ingest.ingest([record()])

        result = ingest.ingest([record("Payment reports now include duplicate rows.")])

        assert result.embedded_by_space == {primary.spec.space_id: 1}
        assert catalog.stale_chunk_ids(primary.spec) == []
        assert len(primary.document_calls) == 2


def test_changed_structured_metadata_refreshes_dense_filter_snapshot() -> None:
    primary = FakeEmbedding(dimension=16)
    with SqliteDocumentCatalog() as catalog:
        ingest, store, _ = pipeline(catalog, primary)
        original = record()
        ingest.ingest([original])
        changed = Record(
            id=original.id,
            semantic_fields=original.semantic_fields,
            structured={**original.structured, "visibility": "public"},
            source=original.source,
        )

        result = ingest.ingest([changed])

        assert result.embedded_by_space == {primary.spec.space_id: 1}
        assert len(primary.document_calls) == 2
        vector = primary.embed_query("reports")
        assert store.search(vector, filter={"visibility": "internal"}) == []
        assert store.search(vector, filter={"visibility": "public"})


def test_reembed_stale_repairs_catalog_change() -> None:
    primary = FakeEmbedding(dimension=16)
    with SqliteDocumentCatalog() as catalog:
        ingest, store, _ = pipeline(catalog, primary)
        ingest.ingest([record()])
        catalog.upsert_chunks(
            [
                CatalogChunk(
                    chunk_id="INC-1104::chunk-0000",
                    doc_id="INC-1104",
                    text="Title: Missing reports\nBody: repaired content",
                )
            ]
        )

        result = ingest.reembed_stale(primary.spec)

        assert result.stale_count == 1
        assert result.embedded_count == 1
        assert catalog.stale_chunk_ids(primary.spec) == []
        assert "repaired content" in store.get(["INC-1104::chunk-0000"])[0].text


def test_reingestion_prunes_chunks_removed_by_chunker() -> None:
    primary = FakeEmbedding(dimension=16)
    config = IngestionConfig(batch_size=2)
    with SqliteDocumentCatalog() as catalog:
        ingest, store, _ = pipeline(
            catalog,
            primary,
            config=config,
            chunker=WordChunker(max_words=5, overlap_words=0),
        )
        first = ingest.ingest([record("one two three four five six seven eight")])
        second = ingest.ingest([record("short")])

        assert first.chunk_count > 1
        assert second.chunk_count == 1
        assert second.removed_chunk_count == first.chunk_count - 1
        assert store.count() == 1


def test_primary_required_rejects_policy_fallback_but_catalog_is_repairable() -> None:
    primary = FakeEmbedding(dimension=16, provider="primary", model="p")
    fallback = FakeEmbedding(dimension=8, provider="fallback", model="f")
    primary_store = NumpyVectorStore(dimension=16)
    fallback_store = NumpyVectorStore(dimension=8)
    with SqliteDocumentCatalog() as catalog:
        ingest = IngestionPipeline(
            catalog,
            {
                primary.spec.space_id: primary_store,
                fallback.spec.space_id: fallback_store,
            },
            EmbeddingRouter(primary, fallback, primary_enabled=False),
        )

        with pytest.raises(PrimaryProviderRequiredError, match="required"):
            ingest.ingest([record()])

        assert catalog.find({"doc_id": "INC-1104"})
        assert catalog.stale_chunk_ids(primary.spec) == ["INC-1104::chunk-0000"]
        assert primary_store.count() == 0
        assert fallback_store.count() == 0


def test_non_required_ingestion_uses_routed_fallback_in_lazy_mode() -> None:
    primary = FakeEmbedding(dimension=16, provider="primary", model="p")
    fallback = FakeEmbedding(dimension=8, provider="fallback", model="f")
    with SqliteDocumentCatalog() as catalog:
        primary_store = NumpyVectorStore(dimension=16)
        fallback_store = NumpyVectorStore(dimension=8)
        ingest = IngestionPipeline(
            catalog,
            {
                primary.spec.space_id: primary_store,
                fallback.spec.space_id: fallback_store,
            },
            EmbeddingRouter(primary, fallback, primary_enabled=False),
            config=IngestionConfig(
                fallback_index=FallbackIndexMode.LAZY,
                ingest_requires_primary=False,
            ),
        )

        result = ingest.ingest([record()])

        assert result.embedded_by_space == {fallback.spec.space_id: 1}
        assert primary_store.count() == 0
        assert fallback_store.count() == 1


def test_pipeline_output_is_hybrid_retrievable_in_fallback_space() -> None:
    primary = FakeEmbedding(dimension=32, provider="primary", model="p")
    fallback = FakeEmbedding(dimension=16, provider="fallback", model="f")
    primary_store = NumpyVectorStore(dimension=32)
    fallback_store = NumpyVectorStore(dimension=16)
    ingest_router = EmbeddingRouter(primary, fallback)
    with SqliteDocumentCatalog() as catalog:
        IngestionPipeline(
            catalog,
            {
                primary.spec.space_id: primary_store,
                fallback.spec.space_id: fallback_store,
            },
            ingest_router,
        ).ingest([record("Incident INC-1104 payment reports are missing.")])
        retriever = build_retriever(
            catalog,
            router=EmbeddingRouter(primary, fallback, primary_enabled=False),
            primary_store=primary_store,
            fallback_store=fallback_store,
        )

        result = retriever.retrieve(
            "INC-1104 payment reports",
            filter={"doc_type": "incident"},
            scope=RetrievalScope(tenant_id="acme", visibility=("internal",)),
        )

        assert result.hits[0].chunk.doc_id == "INC-1104"
        assert result.provider == "fallback"
        assert result.fallback_occurred


def test_primary_failure_is_recorded_and_raised_by_default() -> None:
    primary = FailingEmbedding(dimension=16, provider="primary", model="p")
    with SqliteDocumentCatalog() as catalog:
        ingest, _, _ = pipeline(catalog, primary)

        with pytest.raises(IngestionError, match="primary embedding failed"):
            ingest.ingest([record()])

        assert catalog.stale_chunk_ids(primary.spec) == ["INC-1104::chunk-0000"]


def test_pipeline_rejects_store_space_or_dimension_mismatch() -> None:
    primary = FakeEmbedding(dimension=16)
    catalog = SqliteDocumentCatalog()
    with pytest.raises(ValueError, match="no configured provider"):
        IngestionPipeline(
            catalog,
            {"unknown": NumpyVectorStore()},
            EmbeddingRouter(primary),
        )
    with pytest.raises(ValueError, match="expects 8-dimensional"):
        IngestionPipeline(
            catalog,
            {primary.spec.space_id: NumpyVectorStore(dimension=8)},
            EmbeddingRouter(primary),
        )
    catalog.close()
