from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import openai
import pytest

from vectorstore import OpenAIEmbedding


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
        return SimpleNamespace(data=list(reversed(items)))


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsResource()


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

    vectors = embedder.embed_texts(["a", "bbbb", "cc"])

    assert vectors == [[1.0, 0.0], [4.0, 1.0], [2.0, 0.0]]
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


def test_embedding_empty_input_does_not_call_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    embedder = OpenAIEmbedding(api_key="test-key")

    assert embedder.embed_texts([]) == []
    assert client.embeddings.calls == []
