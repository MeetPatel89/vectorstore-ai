"""Composition helper for the hybrid retriever.

The library stays application-config-agnostic: there is no settings file,
no environment parsing, and no pydantic — applications construct these
arguments from their own configuration system and call
:func:`build_retriever`, which wires the router, per-space stores, analyzer,
and observer into a ready :class:`~vectorstore.hybrid.retriever.Retriever`
with fail-fast validation.
"""

from __future__ import annotations

from vectorstore.catalog.base import DocumentCatalog
from vectorstore.embeddings.base import EmbeddingProvider
from vectorstore.embeddings.policy import BudgetLedger, CircuitBreaker, EmbeddingRouter
from vectorstore.hybrid.query import QueryAnalyzer
from vectorstore.hybrid.retriever import Retriever, RetrieverConfig
from vectorstore.observability.base import RetrievalTraceObserver
from vectorstore.stores.base import VectorStore


def _validate_store(provider: EmbeddingProvider, store: VectorStore, role: str) -> None:
    spec = provider.spec
    if store.dimension is not None and store.dimension != spec.dimension:
        raise ValueError(
            f"{role} store expects {store.dimension}-dimensional vectors but "
            f"embedding space {spec.space_id!r} produces {spec.dimension}"
        )


def build_retriever(
    catalog: DocumentCatalog,
    *,
    primary: EmbeddingProvider | None = None,
    primary_store: VectorStore | None = None,
    fallback: EmbeddingProvider | None = None,
    fallback_store: VectorStore | None = None,
    router: EmbeddingRouter | None = None,
    ledger: BudgetLedger | None = None,
    breaker: CircuitBreaker | None = None,
    analyzer: QueryAnalyzer | None = None,
    observer: RetrievalTraceObserver | None = None,
    config: RetrieverConfig | None = None,
    primary_enabled: bool = True,
    override: str | None = None,
    daily_budget_usd: float | None = None,
    monthly_budget_usd: float | None = None,
) -> Retriever:
    """Compose a :class:`~vectorstore.hybrid.retriever.Retriever`.

    Stores are bound to their provider's embedding space automatically, so
    vectors from different models can never share an index. When ``router``
    is not supplied, one is built from ``primary``/``fallback`` and the
    budget arguments; the catalog itself serves as the durable budget
    ledger when it satisfies the
    :class:`~vectorstore.embeddings.policy.BudgetLedger` protocol and no
    explicit ledger is given.

    Passing neither ``router`` nor ``primary`` yields a retriever with the
    dense branch disabled: lexical and structured retrieval only.
    """
    if router is not None and (primary is not None or fallback is not None):
        raise ValueError("pass either router or primary/fallback providers, not both")

    stores: dict[str, VectorStore] = {}
    if router is None and primary is not None:
        if ledger is None and isinstance(catalog, BudgetLedger):
            ledger = catalog
        router = EmbeddingRouter(
            primary,
            fallback,
            ledger=ledger,
            breaker=breaker,
            primary_enabled=primary_enabled,
            override=override,
            daily_budget_usd=daily_budget_usd,
            monthly_budget_usd=monthly_budget_usd,
        )

    if router is not None:
        pairs: list[tuple[EmbeddingProvider, VectorStore | None, str]] = [
            (router.primary, primary_store, "primary"),
        ]
        if router.fallback is not None:
            pairs.append((router.fallback, fallback_store, "fallback"))
        elif fallback_store is not None:
            raise ValueError("fallback_store given without a fallback provider")
        for provider, store, role in pairs:
            if store is None:
                continue
            _validate_store(provider, store, role)
            stores[provider.spec.space_id] = store

    effective_config = config or RetrieverConfig()
    if router is None and effective_config.dense_enabled:
        effective_config = RetrieverConfig(
            dense_enabled=False,
            lexical_enabled=effective_config.lexical_enabled,
            dense_top_k=effective_config.dense_top_k,
            lexical_top_k=effective_config.lexical_top_k,
            final_top_k=effective_config.final_top_k,
            rrf_k=effective_config.rrf_k,
        )

    return Retriever(
        catalog=catalog,
        stores=stores,
        router=router,
        analyzer=analyzer,
        observer=observer,
        config=effective_config,
    )
