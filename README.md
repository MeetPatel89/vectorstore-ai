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

Every provider declares an `EmbeddingSpec` (provider, model, dimension,
version) identifying its embedding space. A `VectorIndex` is bound to exactly
one space and rejects stores with an incompatible dimension, so vectors from
different embedding models are never compared against one another. Use one
store (collection, table, or directory) per `spec.space_id`.

For ingestion, `Record` separates a source row's semantic fields from its
structured attributes; `semantic_projection()` renders only the semantic
fields into indexable text, and `content_hash()` supports skipping
re-embedding of unchanged content.

## Embedding providers and fallback policy

Two providers ship with the library:

- `OpenAIEmbedding` (primary): OpenAI's embeddings API, default
  `text-embedding-3-small`.
- `SentenceTransformerEmbedding` (fallback, extra `local`): a locally hosted
  Sentence Transformers model, default `all-MiniLM-L6-v2` (384 dimensions,
  L2-normalized). The model loads lazily on first use.

`EmbeddingRouter` deterministically picks one of them per call and returns a
`ProviderSelection` with a machine-readable `SelectionReason`:

1. `manual_override` — `force_primary` / `force_fallback` configuration.
2. `openai_disabled` — primary disabled in configuration.
3. `openai_unavailable` / `openai_rate_limited` — the built-in
   `CircuitBreaker` is open after consecutive failures or during a 429
   backoff window.
4. `budget_daily_exceeded` / `budget_monthly_exceeded` — the `BudgetLedger`
   atomically rejects a reservation whose predicted charge would exceed the
   configured limit.
5. `primary` — otherwise.

```python
from vectorstore import (
    EmbeddingRouter,
    InMemoryBudgetLedger,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
)

router = EmbeddingRouter(
    OpenAIEmbedding(),
    SentenceTransformerEmbedding(),
    ledger=InMemoryBudgetLedger(),
    daily_budget_usd=0.50,
)

selection = router.select("query", texts=["how do I rotate certificates?"])
try:
    embedding = selection.provider.embed_query_with_usage(
        "how do I rotate certificates?"
    )
except Exception:
    if selection.provider is router.primary:
        router.record_failure(selection.reservation)
    raise
vector = embedding.vector
# selection.spec.space_id tells you which vector index to search;
# spaces are never mixed.

if selection.provider is router.primary:
    tokens = (
        embedding.usage.input_tokens
        if embedding.usage is not None
        else selection.provider.estimate_tokens(["how do I rotate certificates?"])
    )
    router.record_usage(tokens=tokens, reservation=selection.reservation)
```

Because the two providers occupy different embedding spaces, keep one vector
store per `spec.space_id` and route queries to the store matching the
selected provider.

OpenAI budget preflight uses the model's `tiktoken` encoding rather than a
character-count approximation. The router atomically reserves the predicted
charge before a call, and successful calls reconcile that reservation with the
embedding API's reported input-token usage; `Retriever` performs this
accounting automatically. Failed calls release their reservation, while stale
reservations expire after five minutes by default. Custom model aliases can supply
`OpenAIEmbedding(..., encoding_name="cl100k_base")`; budgeted routing fails
closed with `TokenCountingUnavailableError` when no model-to-encoding mapping
is available.

Pricing is provider-, model-, and processing-mode-aware. The built-in,
versioned OpenAI catalog covers `text-embedding-3-small`,
`text-embedding-3-large`, and `text-embedding-ada-002`. Unknown prices also
fail closed whenever a budget is enabled. Supply a private or custom rate
without binary floating-point arithmetic:

```python
from vectorstore import EmbeddingPrice, EmbeddingPricing

pricing = EmbeddingPricing(
    (
        EmbeddingPrice.from_usd_per_million(
            "my-provider",
            "my-embedding-model",
            "0.075",
            version="contract-2026-08",
        ),
    )
)
```

Budget limits and charges are converted to integer nanodollars. Ledger audit
rows retain provider, model, processing mode, authoritative tokens, applied
rate, price version, computed charge, and reservation status. The inline
charge is therefore deterministic and auditable; provider invoices can still
be reconciled separately against organization billing data.

## Document catalog (structured find, lexical search, ledgers)

`SqliteDocumentCatalog` (standard library only, zero extra dependencies) is
the system of record for searchable documents. It owns everything except the
dense vectors themselves, which stay in per-space vector stores:

- **Structured retrieval**: `find(filter, scope, limit)` queries documents by
  their natural attributes (`doc_type`, `status`, `tenant_id`, ...) plus any
  custom attributes, using the same filter syntax as vector search. Filters
  are pushed down into SQL (`json_extract` for custom attributes).
- **Lexical retrieval**: `search_lexical(query, k, filter, scope)` uses an
  FTS5 index over chunk text with BM25 ranking, kept in sync with the chunk
  rows by triggers in the same transaction. Raw user queries are sanitized
  into safe MATCH expressions; quoted phrases become phrase queries. Exact
  identifiers like `INC-1104` or `SQLSTATE 23505` are first-class here.
  If FTS5 is unavailable, a typed `LexicalUnavailableError` lets callers
  degrade to dense + structured retrieval.
- **Embedding lifecycle ledger**: `mark_embedded(chunk_id, spec, hash)`
  records which vector exists per (chunk, embedding space);
  `stale_chunk_ids(spec)` returns chunks whose vector is missing or was
  built from outdated content, so re-embedding is incremental.
- **Durable budget ledger**: the catalog satisfies the `BudgetLedger`
  protocol, so it can atomically reserve and reconcile spend across processes.
  Exact nanodollar accounting and complete pricing provenance survive restarts.

Authorization is a `RetrievalScope(tenant_id, visibility)` enforced inside
the SQL of every candidate generator — never by post-filtering:

```python
from vectorstore import (
    CatalogChunk,
    CatalogDocument,
    RetrievalScope,
    SqliteDocumentCatalog,
)

catalog = SqliteDocumentCatalog("corpus.db")
catalog.upsert_documents(
    [
        CatalogDocument(
            doc_id="INC-1104",
            title="Payment reporting data missing",
            doc_type="incident",
            tenant_id="acme",
            visibility="internal",
            status="OPEN",
            attributes={"severity": 3},
        ),
    ]
)
catalog.upsert_chunks(
    [
        CatalogChunk(
            chunk_id="INC-1104:0",
            doc_id="INC-1104",
            text="Incident: INC-1104\nDescription: reconciliation reports are empty",
        ),
    ]
)

scope = RetrievalScope(tenant_id="acme", visibility=("internal", "public"))
open_incidents = catalog.find({"status": "OPEN", "severity": {"$gte": 2}}, scope)
hits = catalog.search_lexical("INC-1104", k=5, scope=scope)
chunks = catalog.get_chunks([hit.chunk_id for hit in hits])
```

Documents without a `tenant_id` are shared across tenants; documents without
a `visibility` label are visible to every scope.

## Hybrid retrieval (the primary API)

`Retriever` orchestrates all three retrieval signals over one catalog and
per-space vector stores, and `build_retriever()` composes it:

```python
from vectorstore import (
    NumpyVectorStore,
    OpenAIEmbedding,
    RetrievalScope,
    SentenceTransformerEmbedding,
    build_retriever,
)

primary = OpenAIEmbedding()
fallback = SentenceTransformerEmbedding()

retriever = build_retriever(
    catalog,  # also serves as the durable budget ledger
    primary=primary,
    primary_store=NumpyVectorStore(),  # 1536-dim space
    fallback=fallback,
    fallback_store=NumpyVectorStore(),  # 384-dim space, never mixed
    daily_budget_usd=0.50,
)

scope = RetrievalScope(tenant_id="acme", visibility=("internal", "public"))
result = retriever.retrieve(
    "reconciliation reports are empty",
    filter={"doc_type": "incident"},
    scope=scope,
)
for hit in result.hits:
    print(hit.score, hit.chunk.chunk_id, hit.dense_rank, hit.lexical_rank)

open_incidents = retriever.find({"status": "OPEN"}, scope)  # structured only
```

One `retrieve()` call runs, in order:

1. **Query analysis** — a deterministic `QueryAnalyzer` (regex, no LLM)
   classifies the query. Identifier queries (`INC-1104`, `SQLSTATE 23505`,
   `ERR_CONNECTION_RESET`) and quoted phrases up-weight the lexical signal;
   ordinary natural-language queries weight both signals equally.
2. **Dense branch** — the `EmbeddingRouter` selects a provider (with a
   reason code), and the query embedding searches only that provider's
   space. If the primary provider fails mid-request, the failure feeds the
   circuit breaker and the fallback space is tried within the same request.
3. **Lexical branch** — `search_lexical` on the catalog, complementary to
   dense retrieval, not a fallback.
4. **Fusion** — weighted Reciprocal Rank Fusion
   (`score = w / (k + rank)`, `k = 60`) over the two ranked ID lists; only
   ranks are combined, never raw cosine/BM25 scores. Ties break by chunk ID
   so results are deterministic. The `rrf()` function is also exported
   directly.

`scope` is enforced inside candidate generation on both branches: SQL for
lexical, metadata-filter pushdown for dense. The dense pushdown is
deliberately conservative — chunks ingested without `tenant_id`/`visibility`
metadata are excluded from scoped dense search rather than treated as
shared.

Retrieval degrades instead of failing while any signal remains: no usable
provider or store means lexical + structured still serve; a missing FTS
index means dense + structured still serve; `retriever.find()` always
works. Every `RetrievalResult` carries full provenance — `query_kind`,
`provider`, `provider_reason`, `fallback_occurred`, `degraded`, per-signal
ranks on each hit, phase timings, and error summaries — which is what the
evaluation plan consumes to compare dense/lexical/hybrid arms.

### Observability

Pass any object with an `on_retrieve(result)` method as
`build_retriever(..., observer=...)` to receive one `RetrievalResult` per
request. The `RetrievalTraceObserver` protocol is dependency-free and the
result carries no query or document text, so observers are content-safe by
default; observer exceptions are swallowed so telemetry can never break
retrieval.

## Install

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required. The core
install ships the NumPy store and OpenAI embeddings:

```bash
uv sync
```

The other backends are optional extras:

```bash
uv sync --extra chroma      # ChromaVectorStore
uv sync --extra faiss       # FaissVectorStore
uv sync --extra azure-sql   # AzureSqlVectorStore (Microsoft's driver)
uv sync --extra local       # SentenceTransformerEmbedding (torch + sentence-transformers)
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

index.index(
    [
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
    ]
)

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

Tests marked `local_model` load the real MiniLM model; they skip
automatically unless the `local` extra is installed.

## Demo

For offline-first, progressive walkthroughs of semantic projection, dense
search, embedding-space safety, provider fallback policy, and the document
catalog through end-to-end hybrid retrieval, see the
[retrieval demos](examples/README.md):

```bash
uv run python examples/01_dense_search.py
uv run python examples/02_provider_routing.py
uv run python examples/03_document_catalog.py
uv run python examples/04_hybrid_retrieval.py
```

The original backend-selection demo below uses OpenAI embeddings.

The demo treats each Markdown file under `data/corpora/nautilus/raw/` as one
chunk, indexes the corpus, and runs sample searches:

```bash
uv run python main.py
```

Chroma is the default (requires the `chroma` extra). To use the in-memory
NumPy backend instead:

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
