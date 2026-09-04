from __future__ import annotations

from types import SimpleNamespace
from typing import cast, override

import openai
import pytest

import vectorstore.embeddings.openai as openai_provider
from vectorstore import (
    EmbeddingResult,
    EmbeddingRouter,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
    TokenCountingUnavailableError,
)


class _FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        inputs = cast(list[str], kwargs["input"])
        items = [
            SimpleNamespace(index=index, embedding=[float(len(text)), float(index)])
            for index, text in enumerate(inputs)
        ]
        usage = SimpleNamespace(
            prompt_tokens=sum(len(text) for text in inputs),
            # Deliberately different: embedding cost accounting is based on
            # billable input/prompt tokens, not a generic total field.
            total_tokens=sum(len(text) for text in inputs) + 100,
        )
        return SimpleNamespace(data=list(reversed(items)), usage=usage)


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsResource()


class _MissingUsageEmbeddingsResource(_FakeEmbeddingsResource):
    @override
    def create(self, **kwargs: object) -> SimpleNamespace:
        response = super().create(**kwargs)
        del response.usage
        return response


def test_openai_embedding_batches_and_restores_response_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    client_options: dict[str, object] = {}

    def make_client(**kwargs: object) -> _FakeClient:
        client_options.update(kwargs)
        return client

    monkeypatch.setattr(openai, "OpenAI", make_client)
    embedder = OpenAIEmbedding(
        api_key="test-key",
        batch_size=2,
        dimensions=2,
    )

    result = embedder.embed_texts_with_usage(["a", "bbbb", "cc"])

    assert result.vectors == [[1.0, 0.0], [4.0, 1.0], [2.0, 0.0]]
    assert result.usage is not None
    assert result.usage.total_tokens == 7
    assert client_options == {"api_key": "test-key", "max_retries": 2}
    assert client.embeddings.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["a", "bbbb"],
            "dimensions": 2,
        },
        {
            "model": "text-embedding-3-small",
            "input": ["cc"],
            "dimensions": 2,
        },
    ]
    assert embedder.dimension == 2


def test_openai_embedding_accepts_an_injected_client_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = _FakeClient()
    embedder = OpenAIEmbedding(client=client, dimensions=2)

    assert embedder.embed_texts(["abc"]) == [[3.0, 0.0]]
    assert client.embeddings.calls[0]["model"] == "text-embedding-3-small"


def test_sentence_transformer_accepts_lazy_injected_model_factory() -> None:
    factory_calls: list[tuple[str, str | None]] = []

    class FakeModel:
        def get_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            assert kwargs["normalize_embeddings"] is True
            return [[float(len(text)), 1.0] for text in texts]

    def model_factory(model: str, device: str | None) -> FakeModel:
        factory_calls.append((model, device))
        return FakeModel()

    embedder = SentenceTransformerEmbedding(
        model="test-model",
        dimension=2,
        device="cpu",
        model_factory=model_factory,
    )

    assert factory_calls == []
    assert embedder.embed_texts(["abc"]) == [[3.0, 1.0]]
    assert factory_calls == [("test-model", "cpu")]


def test_embedding_result_does_not_expose_mutable_vector_state() -> None:
    vectors = [[1.0, 2.0]]
    result = EmbeddingResult(vectors)

    vectors[0][0] = 99.0
    returned = result.vectors
    returned[0][1] = 88.0

    assert result.vector == [1.0, 2.0]


def test_legacy_embedding_method_still_returns_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: _FakeClient())
    embedder = OpenAIEmbedding(api_key="test-key", dimensions=2)

    assert embedder.embed_texts(["abc"]) == [[3.0, 0.0]]


def test_query_embedding_retains_authoritative_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: _FakeClient())
    embedder = OpenAIEmbedding(api_key="test-key", dimensions=2)

    result = embedder.embed_query_with_usage("INC-1104")

    assert result.vector == [8.0, 0.0]
    assert result.usage is not None
    assert result.usage.total_tokens == 8


def test_missing_response_usage_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SimpleNamespace(embeddings=_MissingUsageEmbeddingsResource())
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    embedder = OpenAIEmbedding(api_key="test-key", dimensions=2)

    with pytest.raises(RuntimeError, match="valid token usage"):
        embedder.embed_texts(["abc"])


def test_openai_embedding_uses_known_model_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: _FakeClient())

    assert OpenAIEmbedding(api_key="test-key").dimension == 1536
    assert (
        OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key="test-key",
        ).dimension
        == 3072
    )
    assert (
        OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key="test-key",
        ).dimension
        == 1536
    )


def test_openai_embedding_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedding()


def test_unknown_model_requires_explicit_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: _FakeClient())
    embedder = OpenAIEmbedding(model="custom-model", api_key="test-key")

    with pytest.raises(ValueError, match="pass dimensions explicitly"):
        _ = embedder.dimension


def test_unknown_model_requires_encoding_for_token_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: _FakeClient())
    embedder = OpenAIEmbedding(
        model="custom-model",
        api_key="test-key",
        dimensions=2,
    )

    with pytest.raises(TokenCountingUnavailableError, match="encoding_name"):
        embedder.estimate_tokens(["INC-1104"])


def test_budgeted_unknown_model_fails_closed_without_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: _FakeClient())
    embedder = OpenAIEmbedding(
        model="custom-model",
        api_key="test-key",
        dimensions=2,
    )
    router = EmbeddingRouter(
        embedder,
        daily_budget_usd=1.0,
        cost_per_million_tokens=1.0,
    )

    assert embedder.embed_texts(["abc"]) == [[3.0, 0.0]]
    with pytest.raises(TokenCountingUnavailableError, match="encoding_name"):
        router.select(texts=["INC-1104"])

    assert router.select(estimated_tokens=4).provider is embedder


def test_custom_model_accepts_explicit_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: _FakeClient())
    embedder = OpenAIEmbedding(
        model="custom-model",
        api_key="test-key",
        dimensions=2,
        encoding_name="cl100k_base",
    )

    assert embedder.estimate_tokens(["INC-1104"]) == 4


def test_token_limit_splits_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        openai_provider,
        "token_counts",
        lambda texts, model, encoding_name=None: [8_000] * len(texts),
    )
    embedder = OpenAIEmbedding(api_key="test-key", batch_size=128, dimensions=2)

    result = embedder.embed_texts_with_usage([f"text-{index}" for index in range(38)])
    call_sizes = [
        len(cast(list[str], call["input"])) for call in client.embeddings.calls
    ]

    assert call_sizes == [37, 1]
    assert len(result.vectors) == 38


def test_per_input_token_limit_rejected_before_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        openai_provider,
        "token_counts",
        lambda texts, model, encoding_name=None: [8_193],
    )
    embedder = OpenAIEmbedding(api_key="test-key")

    with pytest.raises(ValueError, match="8193 tokens"):
        embedder.embed_texts(["oversized"])
    assert client.embeddings.calls == []


def test_empty_string_rejected_before_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    embedder = OpenAIEmbedding(api_key="test-key")

    with pytest.raises(ValueError, match="must not be empty"):
        embedder.embed_texts([""])
    assert client.embeddings.calls == []


def test_batch_size_cannot_exceed_api_input_limit() -> None:
    with pytest.raises(ValueError, match="2048"):
        OpenAIEmbedding(api_key="test-key", batch_size=2_049)


def test_embedding_empty_input_does_not_call_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    embedder = OpenAIEmbedding(api_key="test-key")

    assert embedder.embed_texts([]) == []
    result = embedder.embed_texts_with_usage([])
    assert result.vectors == []
    assert result.usage is not None
    assert result.usage.total_tokens == 0
    assert client.embeddings.calls == []
