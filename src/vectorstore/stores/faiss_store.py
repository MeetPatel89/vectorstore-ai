"""In-memory FAISS vector store with directory-based persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self

import numpy as np

from vectorstore.models import Chunk, MetadataFilter, SearchResult, matches

from .base import VectorStore

_FORMAT_VERSION = 1


class FaissVectorStore(VectorStore):
    """An exact cosine-similarity store backed by a FAISS flat index.

    Chunk string IDs are mapped to FAISS integer IDs internally. Vectors are
    L2-normalized before insertion, so inner-product search returns cosine
    similarity scores.
    """

    def __init__(self, dimension: int | None = None) -> None:
        if dimension is not None and dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self._faiss = _import_faiss()
        self._dimension = dimension
        self._index = self._make_index(dimension) if dimension is not None else None
        self._chunks: dict[str, Chunk] = {}
        self._id_to_faiss_id: dict[str, int] = {}
        self._faiss_id_to_id: dict[int, str] = {}
        self._next_faiss_id = 0

    @property
    def dimension(self) -> int | None:
        """The established vector width, if one is known."""

        return self._dimension

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        positions: dict[str, int] = {}
        vector_positions: dict[str, int] = {}
        unique_chunks: list[Chunk] = []
        for position, chunk in enumerate(chunks):
            if not isinstance(chunk.id, str) or not chunk.id:
                raise ValueError("chunk IDs must be non-empty strings")
            existing_position = positions.get(chunk.id)
            if existing_position is None:
                positions[chunk.id] = len(unique_chunks)
                unique_chunks.append(chunk)
            else:
                unique_chunks[existing_position] = chunk
            vector_positions[chunk.id] = position

        normalized = self._prepare_vectors(vectors)
        unique_vectors = np.vstack(
            [normalized[vector_positions[chunk.id]] for chunk in unique_chunks]
        ).astype(np.float32, copy=False)

        faiss_ids: list[int] = []
        replaced_ids: list[int] = []
        for chunk in unique_chunks:
            faiss_id = self._id_to_faiss_id.get(chunk.id)
            if faiss_id is None:
                faiss_id = self._next_faiss_id
                self._next_faiss_id += 1
            else:
                replaced_ids.append(faiss_id)
            faiss_ids.append(faiss_id)

        assert self._index is not None
        if replaced_ids:
            self._index.remove_ids(np.asarray(replaced_ids, dtype=np.int64))
        self._index.add_with_ids(
            unique_vectors,
            np.asarray(faiss_ids, dtype=np.int64),
        )

        for chunk, faiss_id in zip(unique_chunks, faiss_ids, strict=True):
            self._chunks[chunk.id] = chunk
            self._id_to_faiss_id[chunk.id] = faiss_id
            self._faiss_id_to_id[faiss_id] = chunk.id

    def delete(self, ids: list[str]) -> None:
        faiss_ids = [
            self._id_to_faiss_id[id_]
            for id_ in dict.fromkeys(ids)
            if id_ in self._id_to_faiss_id
        ]
        if not faiss_ids:
            return

        assert self._index is not None
        self._index.remove_ids(np.asarray(faiss_ids, dtype=np.int64))
        for faiss_id in faiss_ids:
            id_ = self._faiss_id_to_id.pop(faiss_id)
            del self._id_to_faiss_id[id_]
            del self._chunks[id_]

    def search(
        self,
        vector: list[float],
        k: int = 5,
        filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            return []

        query = self._prepare_query(vector)
        if not self._chunks:
            return []

        if filter is None:
            search_index = self._index
            result_count = min(k, len(self._chunks))
        else:
            eligible_ids = [
                self._id_to_faiss_id[id_]
                for id_, chunk in self._chunks.items()
                if matches(chunk.metadata, filter)
            ]
            if not eligible_ids:
                return []
            search_index = self._filtered_index(eligible_ids)
            result_count = min(k, len(eligible_ids))

        assert search_index is not None
        scores, faiss_ids = search_index.search(query[np.newaxis, :], result_count)
        results = [
            SearchResult(
                chunk=self._chunks[self._faiss_id_to_id[int(faiss_id)]],
                score=float(np.clip(score, -1.0, 1.0)),
            )
            for score, faiss_id in zip(scores[0], faiss_ids[0], strict=True)
            if int(faiss_id) != -1
        ]
        return sorted(results, key=lambda result: (-result.score, result.chunk.id))

    def get(self, ids: list[str]) -> list[Chunk]:
        return [self._chunks[id_] for id_ in ids if id_ in self._chunks]

    def count(self) -> int:
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        """Persist the FAISS index, chunk data, and string-ID mapping."""

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format_version": _FORMAT_VERSION,
            "dimension": self._dimension,
            "next_faiss_id": self._next_faiss_id,
        }
        with (directory / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

        records = [
            {
                "faiss_id": self._id_to_faiss_id[id_],
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for id_, chunk in self._chunks.items()
        ]
        with (directory / "chunks.json").open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)

        index_path = directory / "index.faiss"
        if self._index is None:
            index_path.unlink(missing_ok=True)
        else:
            self._faiss.write_index(self._index, str(index_path))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load and validate a store previously written by :meth:`save`."""

        directory = Path(path)
        try:
            with (directory / "manifest.json").open(encoding="utf-8") as file:
                manifest: Any = json.load(file)
            with (directory / "chunks.json").open(encoding="utf-8") as file:
                records: Any = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid FAISS vector store at {directory}") from exc

        if (
            not isinstance(manifest, dict)
            or manifest.get("format_version") != _FORMAT_VERSION
        ):
            raise ValueError("unsupported FAISS vector store format")
        dimension = manifest.get("dimension")
        if dimension is not None and (not isinstance(dimension, int) or dimension <= 0):
            raise ValueError("stored vector dimension must be positive")
        next_faiss_id = manifest.get("next_faiss_id")
        if not isinstance(next_faiss_id, int) or next_faiss_id < 0:
            raise ValueError("stored next FAISS ID must be a non-negative integer")
        if not isinstance(records, list):
            raise ValueError("stored chunks must be a list")
        if dimension is None and records:
            raise ValueError("stored chunks require a vector dimension")

        store = cls(dimension=dimension)
        chunks: dict[str, Chunk] = {}
        id_to_faiss_id: dict[str, int] = {}
        faiss_id_to_id: dict[int, str] = {}
        try:
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("stored chunk data has invalid types")
                id_ = record["id"]
                text = record["text"]
                metadata = record["metadata"]
                faiss_id = record["faiss_id"]
                if not isinstance(id_, str) or not id_ or not isinstance(text, str):
                    raise ValueError("stored chunk data has invalid types")
                if not isinstance(metadata, dict):
                    raise ValueError("stored chunk data has invalid types")
                if not isinstance(faiss_id, int) or faiss_id < 0:
                    raise ValueError("stored FAISS IDs must be non-negative integers")
                if id_ in chunks or faiss_id in faiss_id_to_id:
                    raise ValueError("stored chunk or FAISS IDs are duplicated")
                chunks[id_] = Chunk(id=id_, text=text, metadata=metadata)
                id_to_faiss_id[id_] = faiss_id
                faiss_id_to_id[faiss_id] = id_
        except KeyError as exc:
            raise ValueError("stored chunk data is missing required fields") from exc

        if faiss_id_to_id and next_faiss_id <= max(faiss_id_to_id):
            raise ValueError("stored next FAISS ID conflicts with existing IDs")

        if dimension is not None:
            try:
                index = store._faiss.read_index(str(directory / "index.faiss"))
            except (OSError, RuntimeError) as exc:
                raise ValueError(f"invalid FAISS vector store at {directory}") from exc
            if (
                index.d != dimension
                or index.ntotal != len(chunks)
                or index.metric_type != store._faiss.METRIC_INNER_PRODUCT
                or not isinstance(index, store._faiss.IndexIDMap2)
            ):
                raise ValueError("stored FAISS index does not match its metadata")
            actual_ids = {
                int(value) for value in store._faiss.vector_to_array(index.id_map)
            }
            if actual_ids != set(faiss_id_to_id):
                raise ValueError("stored FAISS index IDs do not match chunk IDs")
            store._index = index

        store._chunks = chunks
        store._id_to_faiss_id = id_to_faiss_id
        store._faiss_id_to_id = faiss_id_to_id
        store._next_faiss_id = next_faiss_id
        return store

    def _make_index(self, dimension: int) -> Any:
        return self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(dimension))

    def _filtered_index(self, faiss_ids: list[int]) -> Any:
        assert self._index is not None and self._dimension is not None
        index = self._make_index(self._dimension)
        vectors = np.vstack(
            [self._index.reconstruct(faiss_id) for faiss_id in faiss_ids]
        ).astype(np.float32, copy=False)
        index.add_with_ids(vectors, np.asarray(faiss_ids, dtype=np.int64))
        return index

    def _prepare_vectors(self, vectors: list[list[float]]) -> np.ndarray:
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("vectors must form a rectangular numeric matrix") from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(vectors) or matrix.shape[1] == 0:
            raise ValueError("vectors must form a non-empty-width numeric matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("vectors must contain only finite values")

        width = int(matrix.shape[1])
        if self._dimension is None:
            self._dimension = width
            self._index = self._make_index(width)
        elif width != self._dimension:
            raise ValueError(
                f"vector dimension {width} does not match store dimension "
                f"{self._dimension}"
            )
        return _normalize(matrix)

    def _prepare_query(self, vector: list[float]) -> np.ndarray:
        try:
            query = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "query vector must be a one-dimensional numeric vector"
            ) from exc
        if query.ndim != 1 or query.size == 0:
            raise ValueError("query vector must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(query)):
            raise ValueError("query vector must contain only finite values")
        if self._dimension is not None and query.size != self._dimension:
            raise ValueError(
                f"query dimension {query.size} does not match store dimension "
                f"{self._dimension}"
            )
        return _normalize(query[np.newaxis, :])[0]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms != 0,
    )


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError("FaissVectorStore requires faiss-cpu>=1.14.3") from exc
    return faiss
