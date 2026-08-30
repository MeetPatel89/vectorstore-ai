"""Phase 3: structured, lexical, and lifecycle-aware document catalog."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from _corpus import LoadedDocument, load_documents

from vectorstore import (
    CatalogChunk,
    CatalogDocument,
    EmbeddingSpec,
    MetadataFilter,
    RetrievalScope,
    SqliteDocumentCatalog,
)

RUNBOOK_FILTER: MetadataFilter = {"doc_type": "runbook", "status": "active"}
IDENTIFIER_QUERIES = ("INC-1104", "CHG-2407")
DEMO_USAGE_USD = Decimal("0.0002468")
DEMO_USAGE_TOKENS = 12_340
DEMO_SPEC = EmbeddingSpec(
    provider="demo-hash",
    model="blake2b-bow",
    dimension=256,
    version="v1",
)


def main() -> int:
    """Run the Phase 3 document-catalog walkthrough."""
    loaded = load_documents()
    documents = [document for document, _, _ in loaded]
    chunks = [chunk for _, document_chunks, _ in loaded for chunk in document_chunks]

    with TemporaryDirectory(prefix="vectorstore-catalog-") as temp_directory:
        database_path = Path(temp_directory) / "nautilus.sqlite3"
        with SqliteDocumentCatalog(database_path) as catalog:
            catalog.upsert_documents(documents)
            catalog.upsert_chunks(chunks)
            print("\nPHASE 3 — SQLITE DOCUMENT CATALOG")
            print(f"Database:  {database_path}")
            print(f"Documents: {len(documents)}")
            print(f"Chunks:    {len(chunks)}")

            _show_structured_retrieval(catalog)
            _show_lexical_retrieval(catalog, loaded)
            _show_embedding_lifecycle(catalog, chunks)

            catalog.record(
                "openai",
                DEMO_USAGE_TOKENS,
                DEMO_USAGE_USD,
                model="text-embedding-3-small",
                price_version="phase-3-demo-v1",
            )
            spent_before_reopen = catalog.spent_today()

        _show_durable_budget(database_path, spent_before_reopen)

    return 0


def _show_structured_retrieval(catalog: SqliteDocumentCatalog) -> None:
    print("\nPHASE 3 — STRUCTURED RETRIEVAL AND SCOPE")
    unscoped = catalog.find(RUNBOOK_FILTER)
    print("find(doc_type='runbook', status='active'):")
    _print_documents(unscoped)

    internal_scope = RetrievalScope(visibility=("internal",))
    internal = catalog.find(RUNBOOK_FILTER, scope=internal_scope)
    print("\nThe same filter with visibility=('internal',):")
    _print_documents(internal)

    customer_scope = RetrievalScope(visibility=("customer_safe",))
    customer_safe = catalog.find(RUNBOOK_FILTER, scope=customer_scope)
    print("\nThe same filter with visibility=('customer_safe',):")
    _print_documents(customer_safe)
    print("Scope enters find() itself, so SQLite restricts candidates in the query.")


def _show_lexical_retrieval(
    catalog: SqliteDocumentCatalog,
    loaded: list[LoadedDocument],
) -> None:
    print("\nPHASE 3 — FTS5 IDENTIFIER SEARCH")
    titles = {document.doc_id: document.title for document, _, _ in loaded}
    for query in IDENTIFIER_QUERIES:
        print(f"\nsearch_lexical({query!r}):")
        hits = catalog.search_lexical(query, k=5)
        chunks = catalog.get_chunks([hit.chunk_id for hit in hits])
        for hit, chunk in zip(hits, chunks, strict=True):
            section = chunk.section_path or "document"
            print(
                f"  {hit.rank}. {hit.score:.4f}  {chunk.doc_id} / {section}\n"
                f"     {titles[chunk.doc_id]}"
            )


def _show_embedding_lifecycle(
    catalog: SqliteDocumentCatalog,
    chunks: list[CatalogChunk],
) -> None:
    print("\nPHASE 3 — EMBEDDING LIFECYCLE")
    before = catalog.stale_chunk_ids(DEMO_SPEC)
    print(f"Before initial indexing: {len(before)} stale chunks")

    for chunk in chunks:
        if chunk.content_hash is None:
            raise RuntimeError(f"chunk {chunk.chunk_id!r} has no content hash")
        catalog.mark_embedded(chunk.chunk_id, DEMO_SPEC, chunk.content_hash)

    after_indexing = catalog.stale_chunk_ids(DEMO_SPEC)
    print(f"After marking current vectors: {len(after_indexing)} stale chunks")

    original = next(chunk for chunk in chunks if chunk.doc_id == "INC-1104")
    edited = CatalogChunk(
        chunk_id=original.chunk_id,
        doc_id=original.doc_id,
        text=f"{original.text}\n\nDemo edit: metadata refresh verified on every node.",
        chunk_index=original.chunk_index,
        section_path=original.section_path,
        active=original.active,
    )
    catalog.upsert_chunks([edited])
    stale_after_edit = catalog.stale_chunk_ids(DEMO_SPEC)
    if stale_after_edit != [original.chunk_id]:
        raise RuntimeError(
            "expected exactly the edited chunk to become stale, got "
            f"{stale_after_edit!r}"
        )

    print(f"Edited chunk: {original.chunk_id}")
    print(f"  old content_hash: {_short_hash(original.content_hash)}")
    print(f"  new content_hash: {_short_hash(edited.content_hash)}")
    print(f"Stale after edit: {stale_after_edit}")


def _show_durable_budget(database_path: Path, expected: Decimal) -> None:
    print("\nPHASE 3 — DURABLE BUDGET LEDGER")
    print(f"Recorded before close: ${expected}")
    with SqliteDocumentCatalog(database_path) as reopened:
        persisted = reopened.spent_today()
    if persisted != expected:
        raise RuntimeError(f"expected persisted spend {expected}, got {persisted}")
    print(f"Read after reopen:     ${persisted}")
    print("The spend record survived closing and reopening the SQLite catalog.")


def _print_documents(documents: list[CatalogDocument]) -> None:
    if not documents:
        print("  (no matches)")
        return
    for document in documents:
        print(
            f"  {document.doc_id}: {document.title} "
            f"[{document.visibility or 'unlabeled'}]"
        )


def _short_hash(value: str | None) -> str:
    if value is None:
        raise RuntimeError("expected a content hash")
    return f"{value[:16]}..."


if __name__ == "__main__":
    raise SystemExit(main())
