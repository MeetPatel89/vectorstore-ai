# vectorstore-ai

An extensible Python library for ingesting and retrieving structured, dense,
lexical, and hybrid content, with deterministic fusion and embedding-provider
fallback. It separates source adaptation, chunking, embedding, and storage and
ships with four cosine-similarity backends:

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

For ingestion, adapters turn Markdown, CSV, and JSON sources into `Record`s.
`Record` separates semantic fields from structured attributes;
`semantic_projection()` renders only semantic fields into indexable text, and
the lifecycle-aware `IngestionPipeline` skips unchanged vectors by content
hash.

`Record`, `Chunk`, and `CatalogDocument` defensively snapshot their input
mappings and expose them read-only. Stored filtering behavior therefore cannot
change because a caller later mutates an input dictionary.

## Status

The package is pre-1.0 (`0.1.0`). Structured retrieval, SQLite, PostgreSQL,
and Azure SQL full-text catalogs, four dense stores, embedding-provider routing
and budgets, hybrid RRF retrieval, Markdown/CSV/JSON ingestion, section and
generic chunking, embedding lifecycle repair, dependency-free retrieval
observation, and optional OpenTelemetry tracing are implemented. Release and
downstream-application migration remain roadmap work.

## Install

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required. The core
install ships the NumPy store and OpenAI embeddings:

```bash
uv sync
```

The other backends and integrations are optional extras:

```bash
uv sync --extra chroma      # ChromaVectorStore
uv sync --extra faiss       # FaissVectorStore
uv sync --extra azure-sql   # AzureSqlVectorStore + AzureSqlDocumentCatalog
uv sync --extra local       # SentenceTransformerEmbedding (torch + sentence-transformers)
uv sync --extra otel        # OTelRetrievalObserver (API only; app owns SDK/exporters)
uv sync --extra postgres    # PostgresDocumentCatalog (Psycopg 3)
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

## Ingestion

`MarkdownSourceAdapter`, `CsvSourceAdapter`, and `JsonSourceAdapter` normalize
files into `Record` objects. Markdown accepts one file or recursively scans a
directory; CSV is row-per-record; JSON accepts an object, an array, a
`{"records": [...]}` wrapper, JSON Lines, or NDJSON. Directory traversal is
stable and recursive for every adapter.

CSV and JSON adapters infer common ID and semantic field names
case-insensitively. For an application-specific schema, make the boundary
explicit; a semantic mapping is `output label -> source field`:

```python
from vectorstore import CsvSourceAdapter, JsonSourceAdapter

faqs = CsvSourceAdapter(
    id_field="ID",
    semantic_fields={"Question": "question", "Answer": "answer"},
    structured_fields=("category", "status"),
)
events = JsonSourceAdapter(
    id_field="event_id",
    semantic_fields=("title", "description", "resolution"),
)
```

`WholeRecordChunker` keeps each projection intact. `WordChunker` splits generic
text into paragraph-aware overlapping windows. `MarkdownSectionChunker` first
splits on H1/H2 boundaries, keeps deeper headings with their parent section,
then applies the same size/overlap limit while repeating title and section
context.

The pipeline owns catalog writes, per-space dense writes, and lifecycle state:

```python
from vectorstore import (
    EmbeddingRouter,
    IngestionPipeline,
    MarkdownSectionChunker,
    MarkdownSourceAdapter,
    NumpyVectorStore,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
    SqliteDocumentCatalog,
)

catalog = SqliteDocumentCatalog("corpus.db")
primary = OpenAIEmbedding()
fallback = SentenceTransformerEmbedding()
primary_store = NumpyVectorStore(dimension=primary.dimension)
fallback_store = NumpyVectorStore(dimension=fallback.dimension)
router = EmbeddingRouter(primary, fallback, ledger=catalog)

pipeline = IngestionPipeline(
    catalog,
    {
        primary.spec.space_id: primary_store,
        fallback.spec.space_id: fallback_store,
    },
    router,
    chunker=MarkdownSectionChunker(),
)
result = pipeline.ingest_source(MarkdownSourceAdapter(), "docs/")
print(result.document_count, result.chunk_count, result.embedded_by_space)
```

The default `IngestionConfig` uses batches of 100, requires the primary
provider, and eagerly builds the fallback space. `fallback_index="lazy"`
builds that space only after policy actually routes ingestion there (or when
`reembed_stale(fallback.spec)` is called); `"off"` never writes it. Set
`ingest_requires_primary=False` only when a policy-routed fallback-only ingest
is acceptable.

On repeated ingestion, a `(chunk, space)` whose ledger hash still matches is
not embedded again. Changed or missing vectors can be repaired with
`pipeline.reembed_stale(spec)`. Replacing a document also removes chunks no
longer emitted by the chunker from the catalog, lexical index, lifecycle
ledger, and configured vector stores. Catalog rows are written before vector
calls, so a provider or store failure remains visible as stale lifecycle state
and is safe to retry. A structured-metadata change invalidates and rebuilds the
affected dense entries even when text is unchanged, keeping tenant,
visibility, and filter metadata synchronized with the catalog.

## Embedding providers and fallback policy

Two providers ship with the library:

- `OpenAIEmbedding` (primary): OpenAI's embeddings API, default
  `text-embedding-3-small`.
- `SentenceTransformerEmbedding` (fallback, extra `local`): a locally hosted
  Sentence Transformers model, default `all-MiniLM-L6-v2` (384 dimensions,
  L2-normalized). The model loads lazily on first use.

Provider configuration is read-only after construction so an established
`EmbeddingSpec` cannot drift. Applications and tests can inject an
`OpenAIClient` through `OpenAIEmbedding(client=...)` or a
`SentenceTransformerModelFactory` through `model_factory=...`; the default
paths still construct the official SDK client and lazily load the local model.

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

`SqliteDocumentCatalog` (standard library only), `PostgresDocumentCatalog`
(the optional `postgres` extra), and `AzureSqlDocumentCatalog` (the optional
`azure-sql` extra) are interchangeable systems of record for searchable
documents. They own everything except the dense vectors themselves, which stay
in per-space vector stores:

- **Structured retrieval**: `find(filter, scope, limit)` queries documents by
  their natural attributes (`doc_type`, `status`, `tenant_id`, ...) plus any
  custom attributes, using the same filter syntax as vector search. Filters
  are pushed down into SQL (`json_extract` in SQLite and typed `JSONB`
  predicates in PostgreSQL, and typed `OPENJSON` predicates in Azure SQL for
  custom attributes).
- **Lexical retrieval**: `search_lexical(query, k, filter, scope)` uses either
  SQLite FTS5 with BM25 ranking or a PostgreSQL stored `tsvector` with a GIN
  inverted index and `ts_rank_cd` ranking. SQLite sanitizes queries into safe
  MATCH expressions; PostgreSQL parameterizes raw input and converts it with
  `websearch_to_tsquery`, which also preserves quoted-phrase semantics. Azure
  SQL uses a parameterized, safely constructed `CONTAINSTABLE` condition and
  native Full-Text ranks. Exact identifiers like `INC-1104` or
  `SQLSTATE 23505` are first-class here.
  A typed `LexicalUnavailableError` lets callers degrade to dense + structured
  retrieval when the backend's lexical schema is unavailable.
- **Embedding lifecycle ledger**: `mark_embedded(chunk_id, spec, hash)`
  records which vector exists per (chunk, embedding space);
  `stale_chunk_ids(spec)` returns chunks whose vector is missing or was
  built from outdated content, so re-embedding is incremental.
- **Document-level replacement**: `replace_chunks(doc_id, chunks)` retains
  ledger rows for stable chunk IDs and returns superseded IDs for pruning from
  dense stores.
- **Durable budget ledger**: the catalog satisfies the `BudgetLedger`
  protocol, so it can atomically reserve and reconcile spend across processes.
  Exact nanodollar accounting and complete pricing provenance survive restarts.

The public boundaries stay focused: `Retriever` depends only on
`RetrievalCatalog` (`find`, `search_lexical`, and `get_chunks`), while
`DocumentCatalog` adds mutation and embedding-lifecycle operations.
`BudgetLedger` remains a separate protocol. All three bundled catalog facades
compose backend-specific budget components and satisfy both contracts,
preserving the convenient `build_retriever(catalog, ...)` default.

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

For PostgreSQL, schema DDL is explicit so a deployment identity can create the
stored `tsvector`, GIN index, tables, and indexes before runtime credentials
are restricted to DML:

```python
from vectorstore import PostgresDocumentCatalog

catalog = PostgresDocumentCatalog(
    "postgresql://app:password@localhost/vectorstore",
    schema_name="retrieval",
)
catalog.create_schema()  # deployment/bootstrap step
catalog.validate_schema()
```

Omit the constructor connection string to read `POSTGRES_CONNECTIONSTRING`.
The default PostgreSQL text-search configuration is `simple`, which is useful
for technical corpora and identifiers. Pass `text_search_config="english"` (or
another installed configuration) when linguistic stemming is preferred.

Azure SQL uses the same explicit deployment/runtime split. Schema version,
Full-Text catalog name, language LCID, key index, and automatic change tracking
are validated before use:

```python
from vectorstore import AzureSqlDocumentCatalog

catalog = AzureSqlDocumentCatalog(
    "Server=example.database.windows.net;Database=retrieval;...",
    schema_name="dbo",
    fulltext_catalog_name="vectorstore_catalog_fulltext",
    language_lcid=1033,
)
catalog.create_schema()  # deployment identity; Full-Text DDL uses autocommit
catalog.validate_schema()
```

Omit the connection string to read `AZURE_SQL_CONNECTIONSTRING`. Table and
ledger DDL is transactional; Full-Text catalog/index creation is deliberately
run on a separate autocommit connection because
[SQL Server disallows `CREATE FULLTEXT INDEX` inside a user transaction](https://learn.microsoft.com/en-us/sql/t-sql/statements/create-fulltext-index-transact-sql?view=sql-server-ver17).
Runtime identities need DML, `SELECT`, and Full-Text query permissions, not
schema-creation permissions.

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
result does not retain query text. Hydrated hits do carry chunk text for the
application, so custom observers must select safe fields. The bundled OTel
observer emits IDs and operational metadata only; observer exceptions are
swallowed so telemetry can never break retrieval.

Install the `otel` extra and inject a tracer from the provider your application
already configures:

```python
from opentelemetry import trace
from vectorstore import OTelRetrievalObserver, build_retriever

observer = OTelRetrievalObserver(
    trace.get_tracer("my-app.retrieval"),
    attributes={"langfuse.trace.metadata.environment": "production"},
)
retriever = build_retriever(catalog, observer=observer)  # lexical-only example
```

The observer reconstructs a `retrieve` root span with `embeddings <model>`,
`dense.search <backend>`, `lexical.search <backend>`, and `fuse rrf` children.
It emits GenAI embedding attributes, `retrieval.*` provider/scope/rank/count/
latency/cost attributes, result IDs, and `retrieval.fallback`,
`retrieval.degraded`, and `budget.threshold_crossed` events. Exporters,
processors, sampling, and resources remain application-owned. Custom root
attributes pass through, while generated retrieval attributes win collisions.
Error categories and counts are recorded, but backend exception messages are
not exported because they may contain sensitive request or connection details.

## Tests

The default offline suite uses deterministic local embeddings and makes no API
calls:

```bash
uv run pytest -m "not local_model"
```

Tests marked `local_model` load the real MiniLM model; they skip
automatically unless the `local` extra is installed. With that extra and the
model available locally (or with network access for its first download), run
the complete suite with `uv run pytest`.

## Demo

For offline-first, progressive walkthroughs of semantic projection, dense
search, embedding-space safety, provider fallback policy, the document
catalog, hybrid retrieval, and lifecycle-aware ingestion, see the
[retrieval demos](examples/README.md):

```bash
uv run python examples/01_dense_search.py
uv run python examples/02_provider_routing.py
uv run python examples/03_document_catalog.py
uv run python examples/04_hybrid_retrieval.py
uv run python examples/05_ingestion.py
```

The top-level entrypoint runs the same Phase 5 demo. It adapts and
section-chunks the bundled Markdown corpus, eagerly builds separate primary
and fallback NumPy indexes, proves a second pass performs no vector writes,
repairs one simulated stale chunk, and runs hybrid searches:

```bash
uv run python main.py
```

It defaults to the deterministic hash provider and is fully offline. Opt into
a real provider with:

```bash
uv run python main.py --provider openai
uv run python main.py --provider local
```

The OpenAI option requires `OPENAI_API_KEY`; the local option requires the
`local` extra and may download its model the first time it runs.
