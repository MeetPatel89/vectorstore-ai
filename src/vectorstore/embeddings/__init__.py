"""Embedding providers."""

from .base import EmbeddingProvider, EmbeddingSpec
from .openai import OpenAIEmbedding

__all__ = ["EmbeddingProvider", "EmbeddingSpec", "OpenAIEmbedding"]
