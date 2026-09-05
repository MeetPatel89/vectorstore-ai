"""Implementation shared by the top-level and Phase 5 ingestion demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _corpus import add_corpus_argument
from _providers import HashEmbedding, make_embedder

from vectorstore import (
    CatalogChunk,
    EmbeddingRouter,
    IngestionPipeline,
    MarkdownSectionChunker,
    MarkdownSourceAdapter,
    NumpyVectorStore,
    RetrievalScope,
    SqliteDocumentCatalog,
    build_retriever,
)


def parse_args() -> argparse.Namespace:
    """Parse provider selection for the ingestion demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_corpus_argument(parser)
    parser.add_argument(
        "--provider",
        choices=("hash", "openai", "local"),
        default="hash",
        help="primary embedding provider (default: %(default)s, fully offline)",
    )
    parser.add_argument("--k", type=int, default=5, help="results per query")
    return parser.parse_args()


def main() -> int:
    """Ingest the bundled Markdown corpus and exercise the hybrid stack."""
    args = parse_args()
    corpus_root: Path = args.corpus
    if args.k <= 0:
        print("--k must be greater than zero", file=sys.stderr)
        return 2
    try:
        primary = make_embedder(args.provider)
    except (ImportError, ValueError) as exc:
        print(f"Could not create provider {args.provider!r}: {exc}", file=sys.stderr)
        return 2

    fallback = HashEmbedding(dimension=96, model="phase5-fallback")
    primary_store = NumpyVectorStore(dimension=primary.dimension)
    fallback_store = NumpyVectorStore(dimension=fallback.dimension)
    router = EmbeddingRouter(primary, fallback)
    stores = {
        primary.spec.space_id: primary_store,
        fallback.spec.space_id: fallback_store,
    }
    adapter = MarkdownSourceAdapter()
    chunker = MarkdownSectionChunker(max_words=300, overlap_words=30)

    print("PHASE 5 — SOURCE ADAPTER → CHUNKS → CATALOG + DENSE SPACES")
    print(f"  Corpus: {corpus_root}")
    print(f"  Primary space:  {primary.spec.space_id}")
    print(f"  Fallback space: {fallback.spec.space_id}")

    with SqliteDocumentCatalog() as catalog:
        pipeline = IngestionPipeline(
            catalog,
            stores,
            router,
            chunker=chunker,
        )
        try:
            first = pipeline.ingest_source(adapter, corpus_root)
            second = pipeline.ingest_source(adapter, corpus_root)
        except Exception as exc:  # noqa: BLE001 - make provider setup actionable
            print(f"Ingestion failed: {exc}", file=sys.stderr)
            return 2

        print(
            f"  First pass: {first.document_count} documents, "
            f"{first.chunk_count} chunks, {first.embedded_count} vector writes"
        )
        print(
            f"  Second pass: {second.embedded_count} vector writes, "
            f"{second.skipped_embedding_count} current vectors skipped"
        )

        records = list(adapter.iter_records(corpus_root))
        sample = chunker.chunk(records[0])[0]
        catalog.upsert_chunks(
            [
                CatalogChunk(
                    chunk_id=sample.chunk_id,
                    doc_id=sample.doc_id,
                    text=f"{sample.text}\nDemo revision: content changed.",
                    chunk_index=sample.chunk_index,
                    section_path=sample.section_path,
                )
            ]
        )
        stale_before = len(catalog.stale_chunk_ids(primary.spec))
        repaired = pipeline.reembed_stale(primary.spec)
        stale_after = len(catalog.stale_chunk_ids(primary.spec))
        print(
            f"  Stale repair: {stale_before} detected, "
            f"{repaired.embedded_count} re-embedded, {stale_after} remain"
        )

        retriever = build_retriever(
            catalog,
            router=router,
            primary_store=primary_store,
            fallback_store=fallback_store,
        )
        scope = RetrievalScope(visibility=("internal", "customer_safe"))
        for query in (
            "INC-1104",
            "users cannot log in after certificate rotation",
        ):
            result = retriever.retrieve(query, scope=scope, k=args.k)
            print(
                f"\n  Query {query!r}: provider={result.provider}, "
                f"reason={result.provider_reason}, hits={len(result.hits)}"
            )
            for rank, hit in enumerate(result.hits, start=1):
                section = hit.chunk.section_path or "document"
                print(
                    f"    {rank}. {hit.chunk.doc_id} / {section} "
                    f"(dense={hit.dense_rank or '-'}, "
                    f"lexical={hit.lexical_rank or '-'})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
