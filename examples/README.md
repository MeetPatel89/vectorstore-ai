# Retrieval demos

These scripts walk through the retrieval subsystem one implemented phase at a
time using the bundled Nautilus ITSM corpus. They default to a deterministic
hash-based embedder, so they need no API key, network access, or model download.

## Phase 1: dense search and embedding spaces

Run the first demo from the repository root:

```bash
uv run python examples/01_dense_search.py
```

It shows which source fields enter `semantic_projection()`, indexes
section-sized chunks in a `NumpyVectorStore`, prints the provider's
`EmbeddingSpec` and content hashes, runs three dense searches, and deliberately
triggers the dimension-mismatch guard that prevents one store from being used
for incompatible vector widths.

The built-in hash provider is for transparent, repeatable mechanics rather than
production-quality ranking. To opt into a real provider:

```bash
uv run python examples/01_dense_search.py --provider openai
uv sync --extra local
uv run python examples/01_dense_search.py --provider local
```

The OpenAI option requires `OPENAI_API_KEY`. The local option requires the
`local` extra and may download its model the first time it runs.

## Phase 2: provider routing, budget, and fallback

Run the provider-policy walkthrough offline:

```bash
uv run python examples/02_provider_routing.py
```

It composes an `EmbeddingRouter` with a primary provider, a fallback provider,
an `InMemoryBudgetLedger`, and a `CircuitBreaker`. Four isolated scenarios show
the exact `ProviderSelection` reason codes for normal routing, an exhausted
daily budget, an open primary circuit, and a `force_fallback` override. The
demo then indexes the corpus into one store per `EmbeddingSpec.space_id` to
make the vector-space boundary concrete.

For an OpenAI primary, the router's preflight budget decision uses the model's
tokenizer and atomically reserves the predicted nanodollar charge. The hybrid
retriever reconciles successful calls with input-token usage reported by the
embeddings API and releases reservations after failures. Prices are keyed by
provider, model, and processing mode; custom models must supply explicit,
versioned pricing when budgets are enabled.

The default primary and fallback are deterministic hash embedders in different
spaces. A real provider can be used as the primary when its prerequisites are
installed and configured:

```bash
uv run python examples/02_provider_routing.py --provider openai
uv run python examples/02_provider_routing.py --provider local
```

## Phase 3: document catalog

Run the SQLite catalog walkthrough offline:

```bash
uv run python examples/03_document_catalog.py
```

It ingests the full corpus into a temporary `SqliteDocumentCatalog`, combines
structured filters with retrieval scope, and uses FTS5 to search exact
identifiers such as `INC-1104` and `CHG-2407`. It also records the current
content hashes in the embedding lifecycle ledger, edits one chunk, and shows
that only the changed chunk requires re-embedding. Finally, it closes and
reopens the database to demonstrate that embedding spend is durable.

The temporary database is removed when the demo exits. This phase does not
embed text or call an external provider, so it has no API key, network, or
optional dependency requirements.
