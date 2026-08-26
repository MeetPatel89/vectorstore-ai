"""Phase 1: dense search, semantic projection, and embedding spaces."""

from __future__ import annotations

import argparse
import sys

from _corpus import load_documents, to_vector_chunks
from _providers import HashEmbedding, make_embedder

from vectorstore import NumpyVectorStore, VectorIndex, semantic_projection

SAMPLE_QUERIES = (
    "users can't log in after certificate rotation",
    "payment export totals don't match dashboard",
    "business tier API clients receive too many requests",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the dense-search walkthrough."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("hash", "openai", "local"),
        default="hash",
        help="embedding provider (default: %(default)s, fully offline)",
    )
    parser.add_argument("--k", type=_positive_int, default=3, help="hits per query")
    return parser.parse_args()


def main() -> int:
    """Run the Phase 1 dense-search walkthrough."""
    args = parse_args()
    try:
        embedder = make_embedder(args.provider)
    except (ImportError, ValueError) as exc:
        print(
            f"Could not create the {args.provider!r} provider: {exc}", file=sys.stderr
        )
        return 2

    documents = load_documents()
    chunks = to_vector_chunks(documents)
    for item in documents:
        if item[0].doc_id == "INC-1104":
            print("--------------------------------")
            print(item)
            print("--------------------------------")
    _, sample_chunks, sample_record = next(
        item for item in documents if item[0].doc_id == "INC-1104"
    )

    print("PHASE 1 — SEMANTIC PROJECTION")
    print(f"Record: {sample_record.id} ({sample_record.source})")
    print("Structured metadata excluded from the projection:")
    for key in ("doc_id", "doc_type", "owner_group", "visibility", "status"):
        print(f"  {key}: {sample_record.structured[key]}")
    print("\nProjection preview (the document is sectioned before indexing):")
    _print_preview(semantic_projection(sample_record))
    print(f"\nFirst embedded section ({sample_chunks[0].chunk_id}):")
    _print_preview(sample_chunks[0].text)

    store = NumpyVectorStore()
    index = VectorIndex(embedder, store)
    spec = index.spec
    print("\nPHASE 1 — EMBEDDING SPACE")
    print(f"Provider:  {spec.provider}")
    print(f"Model:     {spec.model}")
    print(f"Dimension: {spec.dimension}")
    print(f"Version:   {spec.version}")
    print(f"Space ID:  {spec.space_id}")
    print(f"Indexing {len(chunks)} sections from {len(documents)} documents...")
    index.index(chunks)
    for chunk in sample_chunks[:2]:
        assert chunk.content_hash is not None
        print(f"content_hash[{chunk.chunk_id}] = {chunk.content_hash}")

    print("\nPHASE 1 — DENSE SEARCH")
    for query in SAMPLE_QUERIES:
        print(f"\nQuery: {query!r}")
        for rank, result in enumerate(index.search(query, k=args.k), start=1):
            metadata = result.chunk.metadata
            print(
                f"  {rank}. {result.score:+.4f}  {metadata['doc_id']}"
                f" / {metadata.get('section_path', 'document')}"
            )
            print(f"     {metadata['title']}")

    print("\nPHASE 1 — SPACE-MISMATCH GUARD")
    alternate_dimension = 64 if spec.dimension != 64 else 128
    alternate = HashEmbedding(
        dimension=alternate_dimension,
        model=f"blake2b-bow-{alternate_dimension}",
    )
    alternate_store = NumpyVectorStore()
    alternate_index = VectorIndex(alternate, alternate_store)
    alternate_index.index(chunks)
    print(f"Primary store:   {spec.space_id}")
    print(f"Alternate store: {alternate_index.spec.space_id}")
    try:
        VectorIndex(alternate, store)
    except ValueError as exc:
        print(f"Guard triggered: {exc}")
    else:  # pragma: no cover - this indicates a broken library invariant
        raise RuntimeError("expected VectorIndex to reject a mismatched store")

    print(
        "Each embedding space gets its own store; the alternate provider never "
        "writes into the primary store."
    )
    return 0


def _print_preview(text: str, max_lines: int = 12) -> None:
    lines = text.splitlines()
    for line in lines[:max_lines]:
        print(f"  {line}")
    if len(lines) > max_lines:
        print(f"  ... ({len(lines) - max_lines} more lines)")


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
