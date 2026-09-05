"""Exercise an installed distribution without importing the repository source."""

from __future__ import annotations

from argparse import ArgumentParser
from importlib import import_module
from importlib.metadata import version
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import override

from vectorstore import (
    Chunk,
    EmbeddingProvider,
    EmbeddingRouter,
    EmbeddingSpec,
    IngestionPipeline,
    NumpyVectorStore,
    Record,
    RetrievalScope,
    SqliteDocumentCatalog,
    build_retriever,
)


class SmokeEmbedding(EmbeddingProvider):
    """Small deterministic provider that requires neither a model nor a tokenizer."""

    @property
    @override
    def spec(self) -> EmbeddingSpec:
        """Return the fixed smoke-test embedding space."""
        return EmbeddingSpec("smoke", "keywords", 3)

    @override
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a few sample keywords into deterministic vectors."""
        return [
            [
                float(word in text.lower())
                for word in ("login", "certificate", "payment")
            ]
            for text in texts
        ]


def exercise_core(directory: Path) -> None:
    """Verify persistence, ingestion, lexical search, fusion, and scope."""
    store = NumpyVectorStore()
    store.upsert([Chunk("one", "login")], [[1.0, 0.0, 0.0]])
    store.save(directory / "vectors")
    reopened = NumpyVectorStore.load(directory / "vectors")
    assert reopened.search([1.0, 0.0, 0.0], k=1)[0].chunk.id == "one"
    provider = SmokeEmbedding()
    dense = NumpyVectorStore(dimension=3)
    records = [
        Record(
            "INC-1104",
            {"Body": "INC-1104 login certificate rotation"},
            {"visibility": "customer_safe"},
        ),
        Record(
            "secret",
            {"Body": "login certificate rotation internal"},
            {"visibility": "internal"},
        ),
    ]
    with SqliteDocumentCatalog(directory / "catalog.sqlite3") as catalog:
        pipeline = IngestionPipeline(
            catalog,
            {provider.spec.space_id: dense},
            EmbeddingRouter(provider),
        )
        assert pipeline.ingest(records).document_count == 2
        assert pipeline.ingest(records).embedded_count == 0
        assert catalog.search_lexical("INC-1104", k=1)
        retriever = build_retriever(catalog, primary=provider, primary_store=dense)
        result = retriever.retrieve(
            "login certificate",
            scope=RetrievalScope(
                visibility=("customer_safe",),
            ),
        )
        assert result.hits and {hit.chunk.doc_id for hit in result.hits} == {"INC-1104"}
        assert result.dense_candidates > 0 and result.lexical_candidates > 0
    with SqliteDocumentCatalog(directory / "catalog.sqlite3") as catalog:
        assert len(catalog.find()) == 2


def exercise_extra(extra: str, directory: Path) -> None:
    """Verify each optional extra in its own consumer environment."""
    if extra == "core":
        for module in (
            "chromadb",
            "faiss",
            "torch",
            "sentence_transformers",
            "psycopg",
            "mssql_python",
            "opentelemetry",
            "pytest",
            "dotenv",
        ):
            assert find_spec(module) is None, f"unexpected core dependency: {module}"
        return
    modules = {
        "chroma": "chromadb",
        "faiss": "faiss",
        "otel": "opentelemetry.trace",
        "postgres": "psycopg",
        "azure-sql": "mssql_python",
    }
    import_module(modules[extra])
    if extra in {"chroma", "faiss"}:
        from vectorstore import ChromaVectorStore, FaissVectorStore

        store = (
            ChromaVectorStore(path=directory / "chroma", collection_name="smoke-test")
            if extra == "chroma"
            else FaissVectorStore()
        )
        store.upsert([Chunk("extra", "test")], [[1.0, 0.0]])
        assert store.search([1.0, 0.0], k=1)[0].chunk.id == "extra"
    elif extra == "otel":
        from opentelemetry.trace import get_tracer

        from vectorstore import OTelRetrievalObserver

        OTelRetrievalObserver(get_tracer("vectorstore.smoke"))


def main() -> None:
    """Check installed metadata and public behavior using only declared dependencies."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("expected_version")
    parser.add_argument(
        "--extra",
        choices=("core", "chroma", "faiss", "otel", "postgres", "azure-sql"),
        default="core",
    )
    args = parser.parse_args()
    assert version("vectorstore-ai") == args.expected_version
    assert files("vectorstore").joinpath("py.typed").is_file()
    package = import_module("vectorstore")
    assert package.__file__ is not None
    assert (
        Path(__file__).resolve().parents[1] / "src"
        not in Path(package.__file__).resolve().parents
    )
    with TemporaryDirectory(prefix="vectorstore-smoke-") as temporary:
        directory = Path(temporary)
        exercise_core(directory)
        exercise_extra(args.extra, directory)
    print(f"vectorstore-ai {args.expected_version}: {args.extra} smoke test passed")


if __name__ == "__main__":
    main()
