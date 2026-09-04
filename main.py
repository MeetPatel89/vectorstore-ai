"""Index and search the included Nautilus markdown corpus."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vectorstore import (
    Chunk,
    OpenAIEmbedding,
    VectorIndex,
    create_store,
)

CORPUS_ROOT = Path(__file__).parent / "data" / "corpora" / "nautilus" / "raw"
SAMPLE_QUERIES: tuple[tuple[str, dict[str, object] | None], ...] = (
    ("users can't log in after certificate rotation", None),
    ("payment export totals don't match dashboard", None),
    ("users can't log in after certificate rotation", {"doc_type": "runbooks"}),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the backend-selection demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        choices=("azure-sql", "chroma", "faiss", "numpy"),
        default=os.environ.get("VECTORSTORE_BACKEND", "chroma"),
        help="storage backend (default: %(default)s; env: VECTORSTORE_BACKEND)",
    )
    parser.add_argument(
        "--path",
        default=os.environ.get("VECTORSTORE_PATH", ".chroma"),
        help="Chroma persistence directory (default: %(default)s)",
    )
    parser.add_argument("--k", type=int, default=3, help="results per query")
    return parser.parse_args()


def load_chunks(root: Path = CORPUS_ROOT) -> list[Chunk]:
    """Load Markdown files below *root* as vector-store chunks."""
    chunks: list[Chunk] = []
    for markdown_file in sorted(root.rglob("*.md")):
        chunks.append(
            Chunk(
                id=markdown_file.relative_to(root).as_posix(),
                text=markdown_file.read_text(encoding="utf-8"),
                metadata={"doc_type": markdown_file.parent.name},
            )
        )
    return chunks


def main() -> int:
    """Index the bundled corpus and run the sample searches."""
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Set it before running the "
            "semantic-search demo.",
            file=sys.stderr,
        )
        return 2

    chunks = load_chunks()
    if not chunks:
        print(f"No markdown files found under {CORPUS_ROOT}", file=sys.stderr)
        return 1

    embedder = OpenAIEmbedding()
    if args.store == "chroma":
        store = create_store(
            "chroma",
            path=args.path,
            collection_name="nautilus-demo",
        )
    elif args.store == "azure-sql":
        store = create_store("azure-sql", dimension=embedder.dimension)
    else:
        store = create_store(args.store)

    index = VectorIndex(embedder, store)
    print(f"Indexing {len(chunks)} documents with the {args.store} store...")
    index.index(chunks)

    for query, filter_ in SAMPLE_QUERIES:
        filter_label = f" filter={filter_}" if filter_ else ""
        print(f"\nQuery: {query!r}{filter_label}")
        results = index.search(query, k=args.k, filter=filter_)
        if not results:
            print("  No matching documents")
            continue
        for rank, result in enumerate(results, start=1):
            snippet = " ".join(result.chunk.text.split())[:180]
            print(f"  {rank}. {result.score:+.4f}  {result.chunk.id}\n     {snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
