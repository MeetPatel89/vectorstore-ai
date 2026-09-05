"""Locally hosted Sentence Transformers embedding provider."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

from .base import EmbeddingProvider, EmbeddingSpec

_MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
}

SentenceTransformerModelFactory = Callable[[str, str | None], Any]


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Generate embeddings with a locally hosted Sentence Transformers model.

    Used as the dense-retrieval fallback when the OpenAI provider is
    disabled, unavailable, rate limited, or over budget. Vectors live in
    their own embedding space (provider ``st``) and are never comparable
    with OpenAI vectors.

    The model itself is loaded lazily on first use so that constructing the
    provider (for example inside an
    :class:`~vectorstore.embeddings.policy.EmbeddingRouter`) does not trigger a
    model download.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        dimension: int | None = None,
        batch_size: int = 32,
        device: str | None = None,
        version: str = "v1",
        model_factory: SentenceTransformerModelFactory | None = None,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be greater than zero")
        if dimension is not None and (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise ValueError("dimension must be greater than zero")
        if device is not None and (not isinstance(device, str) or not device):
            raise ValueError("device must be a non-empty string or None")
        if not isinstance(version, str) or not version:
            raise ValueError("version must be a non-empty string")
        if model_factory is not None and not callable(model_factory):
            raise TypeError("model_factory must be callable")

        if dimension is None:
            try:
                dimension = _MODEL_DIMENSIONS[model]
            except KeyError as exc:
                raise ValueError(
                    f"unknown embedding dimension for model {model!r}; "
                    "pass dimension explicitly"
                ) from exc

        self._model_name = model
        self._batch_size = batch_size
        self._device = device
        self._version = version
        self._dimension = dimension
        self._model_factory = (
            model_factory if model_factory is not None else _default_model_factory
        )
        self._loaded_model: Any = None

    @property
    def model(self) -> str:
        """The immutable Sentence Transformers model identifier."""
        return self._model_name

    @property
    def batch_size(self) -> int:
        """The immutable encode batch size."""
        return self._batch_size

    @property
    def device(self) -> str | None:
        """The configured inference device, when one is forced."""
        return self._device

    @property
    def version(self) -> str:
        """The immutable embedding-configuration version."""
        return self._version

    @property
    @override
    def spec(self) -> EmbeddingSpec:
        """Describe the provider's local embedding space."""
        return EmbeddingSpec(
            provider="st",
            model=self.model,
            dimension=self._dimension,
            version=self.version,
        )

    def _model(self) -> Any:
        if self._loaded_model is None:
            self._loaded_model = self._model_factory(self.model, self.device)
            # Renamed in sentence-transformers 6.x; keep the fallback for 5.x.
            probe = getattr(self._loaded_model, "get_embedding_dimension", None)
            if probe is None:
                probe = getattr(self._loaded_model, "get_sentence_embedding_dimension")
            produced = probe()
            if produced is not None and produced != self._dimension:
                raise ValueError(
                    f"model {self.model!r} produces {produced}-dimensional "
                    f"vectors but the provider was configured for {self._dimension}"
                )
        return self._loaded_model

    @override
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts with the locally hosted model."""
        if not texts:
            return []

        # L2-normalized output so cosine similarity equals the dot product,
        # matching how the vector stores score OpenAI embeddings.
        vectors = self._model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]


def _default_model_factory(model: str, device: str | None) -> Any:
    """Load the optional backend only when embeddings are first requested."""
    try:
        # The model extra is exercised in the separate CPU consumer workflow.
        from sentence_transformers import (  # ty: ignore[unresolved-import]
            SentenceTransformer,
        )
    except ImportError as exc:
        raise ImportError(
            "SentenceTransformerEmbedding requires the 'sentence-transformers' "
            "package; install it with uv sync --extra local"
        ) from exc
    return SentenceTransformer(model, device=device)
