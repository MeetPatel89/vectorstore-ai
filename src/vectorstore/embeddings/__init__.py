"""Embedding providers."""

from .base import EmbeddingProvider
from .openai import OpenAIEmbedding

__all__ = ["EmbeddingProvider", "OpenAIEmbedding"]
