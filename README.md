# vectorstore-ai

An extensible Python library for semantic search over pre-chunked text. It
separates embedding from storage and ships with two cosine-similarity backends:

- `NumpyVectorStore`: exact in-memory search with file persistence.
- `ChromaVectorStore`: persistent local search through Chroma.

`VectorIndex` composes either store with an `EmbeddingProvider`. The included
`OpenAIEmbedding` uses OpenAI's embeddings API, while tests can use any local or
fake implementation of the small provider interface.

## Install

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

For OpenAI embeddings, export an API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Quickstart

```python
from vectorstore import Chunk, OpenAIEmbedding, VectorIndex, create_store

store = create_store("numpy")
index = VectorIndex(OpenAIEmbedding(), store)

index.index([
    Chunk(
        id="runbook-sso",
        text="Troubleshoot login failures after signing certificate rotation.",
        metadata={"doc_type": "runbook", "priority": 2},
    ),
    Chunk(
        id="policy-change",
        text="Production certificate changes require approval.",
        metadata={"doc_type": "policy", "priority": 1},
    ),
])

results = index.search(
    "users cannot log in after certificate rotation",
    k=3,
    filter={"doc_type": {"$in": ["runbook", "known_issue"]}},
)
for result in results:
    print(result.score, result.chunk.id)
```

Filters support equality, `$in`, `$gt`, `$gte`, `$lt`, and `$lte`. Conditions
on multiple metadata keys are ANDed together.

To persist a NumPy store explicitly:

```python
from vectorstore import NumpyVectorStore

store.save(".vectors")
store = NumpyVectorStore.load(".vectors")
```

For automatic local persistence, create Chroma with a directory and collection:

```python
store = create_store(
    "chroma",
    path=".chroma",
    collection_name="support-docs",
)
```

## Tests

The full suite uses deterministic local embeddings and makes no API calls:

```bash
uv run pytest
```

## Demo

The demo treats each Markdown file under `data/corpora/nautilus/raw/` as one
chunk, indexes the corpus, and runs sample searches:

```bash
uv run python main.py
```

Chroma is the default. To use the in-memory NumPy backend instead:

```bash
uv run python main.py --store numpy
```

`VECTORSTORE_BACKEND` and `VECTORSTORE_PATH` can also set the backend and Chroma
directory. The demo requires `OPENAI_API_KEY`; the library's offline tests do
not.
