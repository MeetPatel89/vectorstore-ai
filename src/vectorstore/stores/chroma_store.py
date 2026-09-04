"""Persistent Chroma vector store adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

if TYPE_CHECKING:
    from chromadb.base_types import Metadata, PyVector

from vectorstore.models import Chunk, MetadataFilter, MetadataValue, SearchResult

from .base import VectorStore

_SUPPORTED_OPERATORS = frozenset({"$in", "$gt", "$gte", "$lt", "$lte"})


class ChromaVectorStore(VectorStore):
    """A persistent local vector store backed by Chroma."""

    def __init__(
        self,
        path: str | Path = ".chroma",
        collection_name: str = "default",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError(
                "ChromaVectorStore requires the 'chroma' extra "
                "(chromadb>=1.5.5); install it with `uv sync --extra chroma`"
            ) from exc

        self._path = str(path)
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=self.path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def path(self) -> str:
        """The immutable persistence path configured for this store."""
        return self._path

    @property
    def collection_name(self) -> str:
        """The immutable Chroma collection name."""
        return self._collection_name

    @override
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert new chunks and replace existing chunks with matching IDs."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        unique_chunks: list[Chunk] = []
        unique_vectors: list[list[float]] = []
        positions: dict[str, int] = {}
        for chunk, vector in zip(chunks, vectors, strict=True):
            if not isinstance(chunk.id, str) or not chunk.id:
                raise ValueError("chunk IDs must be non-empty strings")
            position = positions.get(chunk.id)
            if position is None:
                positions[chunk.id] = len(unique_chunks)
                unique_chunks.append(chunk)
                unique_vectors.append(vector)
            else:
                unique_chunks[position] = chunk
                unique_vectors[position] = vector

        ids = [chunk.id for chunk in unique_chunks]
        chroma_vectors: list[PyVector] = list(unique_vectors)
        self._collection.upsert(
            ids=ids,
            embeddings=chroma_vectors,
            documents=[chunk.text for chunk in unique_chunks],
            # Chroma accepts None entries to represent absent metadata, but
            # its public type alias currently omits them from metadata lists.
            metadatas=cast(Any, self._replacement_metadatas(ids, unique_chunks)),
        )

    @override
    def delete(self, ids: list[str]) -> None:
        """Delete chunks with the requested IDs when present."""
        if ids:
            self._collection.delete(ids=list(dict.fromkeys(ids)))

    @override
    def search(
        self,
        vector: list[float],
        k: int = 5,
        filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks matching the optional filter."""
        if k <= 0 or self.count() == 0:
            return []

        query: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": min(k, self.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        where = _translate_filter(filter)
        if where is not None:
            query["where"] = where

        response = self._collection.query(**query)
        ids = _first_result_list(response.get("ids"))
        documents = _first_result_list(response.get("documents"))
        metadatas = _first_result_list(response.get("metadatas"))
        distances = _first_result_list(response.get("distances"))

        results: list[SearchResult] = []
        for position, id_ in enumerate(ids):
            document = documents[position] if position < len(documents) else ""
            metadata = metadatas[position] if position < len(metadatas) else None
            distance = distances[position] if position < len(distances) else 1.0
            score = max(-1.0, min(1.0, 1.0 - float(distance)))
            results.append(
                SearchResult(
                    chunk=Chunk(
                        id=str(id_),
                        text=str(document or ""),
                        metadata=_coerce_metadata(metadata),
                    ),
                    score=score,
                )
            )
        return results

    @override
    def get(self, ids: list[str]) -> list[Chunk]:
        """Return known chunks in requested-ID order."""
        if not ids:
            return []

        requested = list(dict.fromkeys(ids))
        response = self._collection.get(
            ids=requested,
            include=["documents", "metadatas"],
        )
        returned_ids = response.get("ids") or []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []
        by_id = {
            str(id_): Chunk(
                id=str(id_),
                text=str(documents[position] or ""),
                metadata=_coerce_metadata(metadatas[position]),
            )
            for position, id_ in enumerate(returned_ids)
        }
        return [by_id[id_] for id_ in ids if id_ in by_id]

    @override
    def count(self) -> int:
        """Return the number of stored chunks."""
        return int(self._collection.count())

    def _replacement_metadatas(
        self, ids: list[str], chunks: list[Chunk]
    ) -> list[Metadata | None]:
        """Build updates that replace rather than merge Chroma metadata."""
        existing = self._collection.get(ids=ids, include=["metadatas"])
        existing_ids = existing.get("ids") or []
        existing_metadatas = existing.get("metadatas") or []
        metadata_by_id = {
            str(id_): dict(existing_metadatas[position] or {})
            for position, id_ in enumerate(existing_ids)
        }

        replacements: list[Metadata | None] = []
        for chunk in chunks:
            replacement: dict[str, MetadataValue | None] = dict(chunk.metadata)
            for old_key in metadata_by_id.get(chunk.id, {}):
                if old_key not in chunk.metadata:
                    # Chroma interprets a None value as deletion of that key.
                    replacement[old_key] = None
            # Chroma rejects an empty metadata dictionary for a new item.
            replacements.append(replacement or None)
        return replacements


def _translate_filter(filter: MetadataFilter | None) -> dict[str, Any] | None:
    """Translate the public filter syntax into a Chroma ``where`` clause."""
    if not filter:
        return None

    clauses: list[dict[str, Any]] = []
    for key, condition in filter.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata filter keys must be non-empty strings")
        if not isinstance(condition, dict):
            _validate_metadata_value(condition, "equality filter")
            clauses.append({key: {"$eq": condition}})
            continue
        if not condition:
            raise ValueError(f"metadata filter for {key!r} cannot be empty")

        for operator, expected in condition.items():
            if operator not in _SUPPORTED_OPERATORS:
                raise ValueError(f"unsupported metadata filter operator: {operator!r}")
            if operator == "$in":
                if not isinstance(expected, (list, tuple, set, frozenset)):
                    raise ValueError("$in requires a list-like value")
                values = list(expected)
                if not values:
                    raise ValueError("$in requires at least one value")
                for value in values:
                    _validate_metadata_value(value, "$in filter")
                clauses.append({key: {operator: values}})
            else:
                if isinstance(expected, bool) or not isinstance(expected, (int, float)):
                    raise ValueError(f"{operator} requires a numeric value")
                clauses.append({key: {operator: expected}})

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _validate_metadata_value(value: object, label: str) -> MetadataValue:
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{label} requires a scalar metadata value")
    return value


def _coerce_metadata(value: object) -> dict[str, MetadataValue]:
    """Validate scalar metadata returned by Chroma."""
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _validate_metadata_value(item, "stored metadata")
        for key, item in value.items()
        if item is not None
    }


def _first_result_list(value: Any) -> list[Any]:
    if value is None or len(value) == 0:
        return []
    return list(value[0] or [])
