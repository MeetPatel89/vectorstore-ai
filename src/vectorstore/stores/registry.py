"""Named vector-store factory registry."""

from collections.abc import Callable
from typing import Any

from .base import VectorStore
from .chroma_store import ChromaVectorStore
from .faiss_store import FaissVectorStore
from .numpy_store import NumpyVectorStore

StoreFactory = Callable[..., VectorStore]

_REGISTRY: dict[str, StoreFactory] = {}


def register_store(name: str, factory: StoreFactory) -> None:
    """Register or replace a named store factory."""

    if not isinstance(name, str) or not name:
        raise ValueError("store name must be a non-empty string")
    if not callable(factory):
        raise TypeError("store factory must be callable")
    _REGISTRY[name] = factory


def create_store(name: str, **kwargs: Any) -> VectorStore:
    """Create a registered store, forwarding backend-specific arguments."""

    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"unknown vector store {name!r}; available stores: {available}"
        ) from exc
    return factory(**kwargs)


register_store("numpy", NumpyVectorStore)
register_store("chroma", ChromaVectorStore)
register_store("faiss", FaissVectorStore)
