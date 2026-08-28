"""Model-aware token counting for embedding requests."""

from __future__ import annotations

from functools import lru_cache

import tiktoken
from tiktoken import Encoding

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class TokenCountingUnavailableError(ValueError):
    """No tokenizer mapping is available for an embedding model."""

    def __init__(self, model: str, encoding_name: str | None = None) -> None:
        if encoding_name is None:
            detail = (
                f"no tiktoken encoding is known for embedding model {model!r}; "
                "pass encoding_name to OpenAIEmbedding or estimated_tokens "
                "to EmbeddingRouter.select"
            )
        else:
            detail = (
                f"tiktoken encoding {encoding_name!r} is unavailable for "
                f"embedding model {model!r}"
            )
        super().__init__(detail)
        self.model = model
        self.encoding_name = encoding_name


def estimate_tokens(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    *,
    encoding_name: str | None = None,
) -> int:
    """Count tokens locally with the encoding used by an embedding model.

    The historical name is retained for API compatibility. This is an exact
    tokenizer count for known model mappings; the provider response remains
    authoritative for billing.
    """
    return sum(token_counts(texts, model, encoding_name=encoding_name))


def token_counts(
    texts: list[str],
    model: str,
    *,
    encoding_name: str | None = None,
) -> list[int]:
    """Return one model-aware token count per input text."""
    encoding = _resolve_encoding(model, encoding_name)
    return [len(encoding.encode(text, disallowed_special=())) for text in texts]


@lru_cache(maxsize=None)
def _resolve_encoding(model: str, encoding_name: str | None) -> Encoding:
    try:
        if encoding_name is not None:
            return tiktoken.get_encoding(encoding_name)
        return tiktoken.encoding_for_model(model)
    except (KeyError, ValueError) as exc:
        raise TokenCountingUnavailableError(model, encoding_name) from exc
