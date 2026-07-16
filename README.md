# vectorstore-ai

An extensible Python library for semantic search over pre-chunked text. It
separates embedding from storage and ships with four cosine-similarity backends:

- `NumpyVectorStore`: exact in-memory search with file persistence.
- `FaissVectorStore`: exact FAISS search with file persistence.
- `ChromaVectorStore`: persistent local search through Chroma.
- `AzureSqlVectorStore`: exact server-side search with Azure SQL native vectors.

`VectorIndex` composes any store with an `EmbeddingProvider`. The included
`OpenAIEmbedding` uses OpenAI's embeddings API, while tests can use any local or
fake implementation of the small provider interface.

## Install

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

Azure SQL support uses Microsoft's optional driver:

```bash
uv sync --extra azure-sql
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

FAISS uses the same explicit persistence pattern and stores a native FAISS
index alongside its chunk data and string-ID mapping:

```python
from vectorstore import FaissVectorStore

store = create_store("faiss")
# ...index chunks through VectorIndex...
store.save(".faiss")
store = FaissVectorStore.load(".faiss")
```

For automatic local persistence, create Chroma with a directory and collection:

```python
store = create_store(
    "chroma",
    path=".chroma",
    collection_name="support-docs",
)
```

For Azure SQL, use a passwordless Microsoft Entra connection string and the
embedding dimension used by your provider:

```bash
export AZURE_SQL_CONNECTIONSTRING="Server=<server>.database.windows.net;Database=<database>;Authentication=ActiveDirectoryDefault;Encrypt=yes;TrustServerCertificate=no;"
```

```python
from vectorstore import AzureSqlVectorStore

store = AzureSqlVectorStore(dimension=1536)
store.validate_schema()
```

Schema creation is deliberately separate from runtime access. See
[Azure SQL setup](docs/AZURE_SQL.md) for table bootstrap, managed identity,
least-privilege grants, firewall/private endpoint configuration, and production
connection strings.

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

Or run the same demo with FAISS:

```bash
uv run python main.py --store faiss
```

Once the Azure SQL table and connection environment variable are configured:

```bash
uv run python main.py --store azure-sql
```

`VECTORSTORE_BACKEND` and `VECTORSTORE_PATH` can also set the backend and Chroma
directory. `AZURE_SQL_CONNECTIONSTRING` configures the Azure SQL backend. The
demo requires `OPENAI_API_KEY`; the library's offline tests do not.
