"""Phase 4: hybrid retrieval, provenance, scope, and graceful degradation."""

from __future__ import annotations

import argparse
import sys
from typing import Never, override

from _corpus import load_documents, to_vector_chunks
from _providers import HashEmbedding, make_embedder

from vectorstore import (
    Chunk,
    EmbeddingProvider,
    EmbeddingSpec,
    NumpyVectorStore,
    QueryAnalyzer,
    QueryKind,
    RetrievalResult,
    RetrievalScope,
    RetrievalTimings,
    RetrievalTraceObserver,
    Retriever,
    RetrieverConfig,
    SqliteDocumentCatalog,
    VectorIndex,
    build_retriever,
)

NATURAL_QUERY = "login failures after certificate rotation"
IDENTIFIER_QUERY = "INC-1104"
DISPLAY_HITS = 6
CONFIG = RetrieverConfig(final_top_k=100)


class PrintObserver(RetrievalTraceObserver):
    """Print content-safe provenance emitted after every retrieval."""

    @override
    def on_retrieve(self, result: RetrievalResult) -> None:
        """Render one retrieval trace without query or document text."""
        provider = (
            f"{result.provider}/{result.model}"
            if result.provider is not None
            else "none"
        )
        print("  Trace:")
        print(
            f"    id={result.query_id[:12]}  kind={result.query_kind}  "
            f"fusion={result.fusion_method}"
        )
        print(
            f"    weights: dense={result.dense_weight:.1f}, "
            f"lexical={result.lexical_weight:.1f}"
        )
        print(
            f"    provider={provider}  reason={result.provider_reason or 'none'}  "
            f"fallback={result.fallback_occurred}"
        )
        print(
            f"    candidates: dense={result.dense_candidates}, "
            f"lexical={result.lexical_candidates}, fused={len(result.hits)}"
        )
        print(
            f"    degraded={result.degraded}  timings={_format_timings(result.timings)}"
        )
        for error in result.errors:
            print(f"    error: {error}")


class FailingQueryEmbedding(EmbeddingProvider):
    """A provider with a valid space whose embedding calls always fail."""

    def __init__(self, spec: EmbeddingSpec) -> None:
        self._spec = spec

    @property
    @override
    def spec(self) -> EmbeddingSpec:
        """Reuse the populated primary store's embedding-space identity."""
        return self._spec

    @override
    def embed_texts(self, texts: list[str]) -> Never:
        """Simulate a provider outage for document embedding."""
        raise RuntimeError("simulated provider outage")

    @override
    def embed_query(self, text: str) -> Never:
        """Simulate a provider outage for query embedding."""
        raise RuntimeError("simulated provider outage")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the hybrid-retrieval walkthrough."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("hash", "openai", "local"),
        default="hash",
        help="primary embedding provider (default: %(default)s, fully offline)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Phase 4 hybrid-retrieval walkthrough."""
    args = parse_args()
    try:
        primary = make_embedder(args.provider)
    except (ImportError, ValueError) as exc:
        print(
            f"Could not create the {args.provider!r} provider: {exc}", file=sys.stderr
        )
        return 2

    fallback = HashEmbedding(dimension=96, model="blake2b-bow-fallback")
    loaded = load_documents()
    catalog_documents = [document for document, _, _ in loaded]
    catalog_chunks = [
        chunk for _, document_chunks, _ in loaded for chunk in document_chunks
    ]
    vector_chunks = to_vector_chunks(loaded)

    print("\nPHASE 4 — COMPOSITION")
    try:
        primary_store = _build_store("Primary", primary, vector_chunks)
        fallback_store = _build_store("Fallback", fallback, vector_chunks)
    except Exception as exc:  # noqa: BLE001 - make provider setup actionable
        print(f"Could not build the dense indexes: {exc}", file=sys.stderr)
        return 2

    observer = PrintObserver()
    analyzer = QueryAnalyzer()
    titles = {document.doc_id: document.title for document in catalog_documents}

    with SqliteDocumentCatalog() as catalog:
        catalog.upsert_documents(catalog_documents)
        catalog.upsert_chunks(catalog_chunks)
        retriever = build_retriever(
            catalog,
            primary=primary,
            primary_store=primary_store,
            fallback=fallback,
            fallback_store=fallback_store,
            analyzer=analyzer,
            observer=observer,
            config=CONFIG,
        )
        print(
            "  build_retriever() connected the catalog, analyzer, router, "
            "observer, and one store per embedding space."
        )

        natural = _run_scenario(
            "1. Natural-language query: dense + lexical + RRF",
            retriever,
            NATURAL_QUERY,
            titles,
        )
        if natural.dense_candidates == 0 or natural.lexical_candidates == 0:
            raise RuntimeError("natural query should receive both retrieval signals")

        identifier = _run_scenario(
            "2. Identifier query: lexical weight is boosted",
            retriever,
            IDENTIFIER_QUERY,
            titles,
        )
        if identifier.query_kind is not QueryKind.IDENTIFIER:
            raise RuntimeError("expected the analyzer to classify INC-1104")
        if args.provider == "hash" and (
            not identifier.hits or identifier.hits[0].chunk.doc_id != "INC-1104"
        ):
            raise RuntimeError("expected the exact-match document to rank first")

        scoped = _run_scenario(
            "3. The same natural query with customer-safe scope",
            retriever,
            NATURAL_QUERY,
            titles,
            scope=RetrievalScope(visibility=("customer_safe",)),
        )
        if len(scoped.hits) >= len(natural.hits):
            raise RuntimeError("expected restrictive scope to return fewer hits")
        print(
            f"  Scope reduced the fused result set from {len(natural.hits)} "
            f"to {len(scoped.hits)} chunks by constraining candidate generation."
        )

        degraded_retriever = build_retriever(
            catalog,
            primary=FailingQueryEmbedding(primary.spec),
            primary_store=primary_store,
            analyzer=analyzer,
            observer=observer,
            config=CONFIG,
        )
        degraded = _run_scenario(
            "4. Primary outage: lexical retrieval still succeeds",
            degraded_retriever,
            NATURAL_QUERY,
            titles,
        )
        if not degraded.degraded or not degraded.errors or not degraded.hits:
            raise RuntimeError("expected a successful, degraded lexical result")

    return 0


def _build_store(
    label: str,
    provider: EmbeddingProvider,
    chunks: list[Chunk],
) -> NumpyVectorStore:
    store = NumpyVectorStore(dimension=provider.spec.dimension)
    index = VectorIndex(provider, store)
    index.index(chunks)
    print(
        f"  {label:8} {provider.spec.space_id}: "
        f"{store.count()} chunks, {store.dimension} dimensions"
    )
    return store


def _run_scenario(
    label: str,
    retriever: Retriever,
    query: str,
    titles: dict[str, str | None],
    *,
    scope: RetrievalScope | None = None,
) -> RetrievalResult:
    print(f"\nPHASE 4 — {label}")
    print(f"  Query: {query!r}")
    if scope is not None:
        print(f"  Scope: visibility={scope.visibility}")
    result = retriever.retrieve(query, scope=scope)
    _print_hits(result, titles)
    return result


def _print_hits(
    result: RetrievalResult,
    titles: dict[str, str | None],
) -> None:
    print("  Top fused hits:")
    print("    #  RRF score  dense  lexical  document / section")
    for rank, hit in enumerate(result.hits[:DISPLAY_HITS], start=1):
        dense_rank = _format_rank(hit.dense_rank)
        lexical_rank = _format_rank(hit.lexical_rank)
        section = hit.chunk.section_path or "document"
        print(
            f"    {rank:<2} {hit.score:>9.6f}  {dense_rank:>5}  "
            f"{lexical_rank:>7}  {hit.chunk.doc_id} / {section}"
        )
        title = titles.get(hit.chunk.doc_id)
        if title is not None:
            print(f"                              {title}")


def _format_rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _format_timings(timings: RetrievalTimings) -> str:
    dense = "skipped" if timings.dense_ms is None else f"{timings.dense_ms:.2f}ms"
    lexical = "skipped" if timings.lexical_ms is None else f"{timings.lexical_ms:.2f}ms"
    return f"dense {dense}, lexical {lexical}, total {timings.total_ms:.2f}ms"


if __name__ == "__main__":
    raise SystemExit(main())
