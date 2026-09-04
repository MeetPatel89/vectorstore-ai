"""In-memory NumPy vector store with directory-based persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self, override

import numpy as np

from vectorstore.models import Chunk, MetadataFilter, SearchResult, matches

from .base import VectorStore


class NumpyVectorStore(VectorStore):
    """A compact exact-search store backed by an in-memory NumPy matrix."""

    def __init__(self, dimension: int | None = None) -> None:
        if dimension is not None and dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self._dimension = dimension
        width = dimension if dimension is not None else 0
        self._vectors = np.empty((0, width), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._id_to_row: dict[str, int] = {}

    @property
    @override
    def dimension(self) -> int | None:
        """The established vector width, if one is known."""
        return self._dimension

    @override
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert new chunks and replace existing chunks with matching IDs."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        for chunk in chunks:
            if not isinstance(chunk.id, str) or not chunk.id:
                raise ValueError("chunk IDs must be non-empty strings")

        normalized = self._prepare_vectors(vectors)
        for chunk, vector in zip(chunks, normalized, strict=True):
            row = self._id_to_row.get(chunk.id)
            if row is None:
                self._id_to_row[chunk.id] = len(self._chunks)
                self._chunks.append(chunk)
                self._vectors = np.vstack((self._vectors, vector[np.newaxis, :]))
            else:
                self._chunks[row] = chunk
                self._vectors[row] = vector

    @override
    def delete(self, ids: list[str]) -> None:
        """Delete chunks with the requested IDs when present."""
        if not ids or not self._chunks:
            return

        removed = set(ids)
        keep_rows = [
            row for row, chunk in enumerate(self._chunks) if chunk.id not in removed
        ]
        if len(keep_rows) == len(self._chunks):
            return

        self._chunks = [self._chunks[row] for row in keep_rows]
        if keep_rows:
            self._vectors = self._vectors[np.asarray(keep_rows, dtype=np.intp)].copy()
        else:
            width = self._dimension if self._dimension is not None else 0
            self._vectors = np.empty((0, width), dtype=np.float32)
        self._rebuild_id_map()

    @override
    def search(
        self,
        vector: list[float],
        k: int = 5,
        filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks matching the optional filter."""
        if k <= 0:
            return []

        query = self._prepare_query(vector)
        if not self._chunks:
            return []

        if filter is None:
            eligible = np.arange(len(self._chunks), dtype=np.intp)
        else:
            eligible = np.fromiter(
                (
                    row
                    for row, chunk in enumerate(self._chunks)
                    if matches(chunk.metadata, filter)
                ),
                dtype=np.intp,
            )
        if eligible.size == 0:
            return []

        scores = self._vectors @ query
        result_count = min(k, int(eligible.size))
        eligible_scores = scores[eligible]
        if result_count < eligible.size:
            top_positions = np.argpartition(-eligible_scores, result_count - 1)[
                :result_count
            ]
            selected = eligible[top_positions]
        else:
            selected = eligible

        ordered_rows = sorted(
            selected.tolist(),
            key=lambda row: (-float(scores[row]), self._chunks[row].id),
        )
        return [
            SearchResult(
                chunk=self._chunks[row],
                score=float(np.clip(scores[row], -1.0, 1.0)),
            )
            for row in ordered_rows
        ]

    @override
    def get(self, ids: list[str]) -> list[Chunk]:
        """Return known chunks in requested-ID order."""
        return [
            self._chunks[self._id_to_row[id_]] for id_ in ids if id_ in self._id_to_row
        ]

    @override
    def count(self) -> int:
        """Return the number of stored chunks."""
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        """Persist this store to ``vectors.npz`` and ``chunks.json``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        ids = np.asarray([chunk.id for chunk in self._chunks], dtype=np.str_)
        dimension = self._dimension if self._dimension is not None else -1
        np.savez_compressed(
            directory / "vectors.npz",
            vectors=self._vectors,
            ids=ids,
            dimension=np.asarray(dimension, dtype=np.int64),
        )
        chunk_records = [
            {"id": chunk.id, "text": chunk.text, "metadata": dict(chunk.metadata)}
            for chunk in self._chunks
        ]
        with (directory / "chunks.json").open("w", encoding="utf-8") as file:
            json.dump(chunk_records, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load and validate a store previously written by :meth:`save`."""
        directory = Path(path)
        try:
            with np.load(directory / "vectors.npz", allow_pickle=False) as archive:
                vectors = np.asarray(archive["vectors"], dtype=np.float32)
                raw_ids = np.asarray(archive["ids"])
                stored_dimension = int(np.asarray(archive["dimension"]).item())
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(f"invalid NumPy vector store at {directory}") from exc

        try:
            with (directory / "chunks.json").open(encoding="utf-8") as file:
                records: Any = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid NumPy vector store at {directory}") from exc

        if stored_dimension == -1:
            dimension: int | None = None
        elif stored_dimension > 0:
            dimension = stored_dimension
        else:
            raise ValueError("stored vector dimension must be positive")

        expected_width = dimension if dimension is not None else 0
        if vectors.ndim != 2 or vectors.shape[1] != expected_width:
            raise ValueError("stored matrix does not match the saved vector dimension")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("stored vectors must contain only finite values")
        if raw_ids.ndim != 1:
            raise ValueError("stored IDs must be a one-dimensional array")

        ids = [str(id_) for id_ in raw_ids.tolist()]
        if not isinstance(records, list) or len(records) != len(ids):
            raise ValueError("stored chunks and vectors have different lengths")
        if vectors.shape[0] != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("stored vector IDs are invalid")

        chunks: list[Chunk] = []
        try:
            for id_, record in zip(ids, records, strict=True):
                if not isinstance(record, dict) or record.get("id") != id_:
                    raise ValueError("stored chunk IDs do not match vector IDs")
                text = record["text"]
                metadata = record["metadata"]
                if not isinstance(text, str) or not isinstance(metadata, dict):
                    raise ValueError("stored chunk data has invalid types")
                chunks.append(Chunk(id=id_, text=text, metadata=metadata))
        except KeyError as exc:
            raise ValueError("stored chunk data is missing required fields") from exc

        store = cls(dimension=dimension)
        store._vectors = vectors.copy()
        store._chunks = chunks
        store._rebuild_id_map()
        return store

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
            if self._vectors.shape[0] == 0:
                self._vectors = np.empty((0, width), dtype=np.float32)
        elif width != self._dimension:
            raise ValueError(
                f"vector dimension {width} does not match store dimension "
                f"{self._dimension}"
            )

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix, dtype=np.float32),
            where=norms != 0,
        )

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

        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return np.zeros_like(query, dtype=np.float32)
        return query / norm

    def _rebuild_id_map(self) -> None:
        self._id_to_row = {chunk.id: row for row, chunk in enumerate(self._chunks)}
