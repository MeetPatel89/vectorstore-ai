"""The hybrid retrieval facade: structured + dense + lexical + fusion.

``Retriever`` is the primary API applications consume. It is deliberately
thin orchestration over injected components — the catalog (structured and
lexical), per-space vector stores (dense), the embedding router (provider
policy), the query analyzer (fusion weights), and an observer. Each concern
stays behind its own seam; the facade only sequences them and implements
graceful degradation, so retrieval keeps working while any signal remains.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

from vectorstore.catalog.base import (
    CatalogChunk,
    CatalogDocument,
    DocumentCatalog,
    LexicalUnavailableError,
    RankedHit,
    RetrievalScope,
)
from vectorstore.embeddings.policy import (
    EmbeddingRouter,
    NoProviderAvailableError,
    ProviderSelection,
    SelectionReason,
    estimate_tokens,
)
from vectorstore.models import MetadataFilter, SearchResult
from vectorstore.observability.base import RetrievalTraceObserver
from vectorstore.stores.base import VectorStore

from .fusion import DEFAULT_RRF_K, rrf
from .query import QueryAnalyzer, QueryKind, QueryProfile


@dataclass(frozen=True)
class RetrieverConfig:
    """Tuning knobs for one retriever instance.

    Defaults follow the plan: 50 candidates from each signal, fused down to
    a final 10 with the standard RRF constant of 60.
    """

    dense_enabled: bool = True
    lexical_enabled: bool = True
    dense_top_k: int = 50
    lexical_top_k: int = 50
    final_top_k: int = 10
    rrf_k: int = DEFAULT_RRF_K

    def __post_init__(self) -> None:
        for label in ("dense_top_k", "lexical_top_k", "final_top_k", "rrf_k"):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True)
class RetrievalHit:
    """One fused result with per-signal provenance.

    ``dense_rank``/``lexical_rank`` are 1-based positions within each
    signal's candidate list (``None`` when the signal did not return the
    chunk). Raw per-signal scores are diagnostics only; they live on
    incomparable scales and are never combined.
    """

    chunk: CatalogChunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None


@dataclass(frozen=True)
class RetrievalTimings:
    """Wall-clock phase latencies in milliseconds (None = phase skipped)."""

    dense_ms: float | None
    lexical_ms: float | None
    total_ms: float


@dataclass(frozen=True)
class RetrievalResult:
    """Everything an application or evaluator needs to know about one request.

    Carries provenance (provider, reason, fallback), degradation state, and
    per-signal ranks on every hit — the seams the evaluation plan relies on
    — without any query or document text.
    """

    hits: tuple[RetrievalHit, ...]
    query_id: str
    query_kind: QueryKind
    provider: str | None
    model: str | None
    space_id: str | None
    provider_reason: str | None
    fallback_occurred: bool
    degraded: bool
    dense_candidates: int
    lexical_candidates: int
    dense_weight: float
    lexical_weight: float
    timings: RetrievalTimings
    fusion_method: str = "rrf"
    errors: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class _DenseOutcome:
    results: list[SearchResult]
    selection: ProviderSelection | None
    fallback_occurred: bool
    attempted: bool
    errors: tuple[str, ...]


def merge_scope_filter(
    filter: MetadataFilter | None, scope: RetrievalScope | None
) -> MetadataFilter | None:
    """Merge a retrieval scope into a metadata filter for dense pushdown.

    The catalog enforces scope in SQL; dense vector stores only understand
    metadata filters, so scope becomes equality/``$in`` conditions on the
    ``tenant_id`` and ``visibility`` metadata keys. This is deliberately
    stricter than the catalog's semantics: chunks ingested without those
    metadata keys are excluded from scoped dense search rather than treated
    as shared, so a scoped request can never widen through the dense path.
    """

    if scope is None or (scope.tenant_id is None and scope.visibility is None):
        return filter
    merged: MetadataFilter = dict(filter or {})
    if scope.tenant_id is not None:
        merged["tenant_id"] = scope.tenant_id
    if scope.visibility is not None:
        merged["visibility"] = {"$in": list(scope.visibility)}
    return merged


class Retriever:
    """Hybrid retrieval over one document catalog and per-space vector stores.

    ``stores`` maps :attr:`~vectorstore.embeddings.base.EmbeddingSpec.space_id`
    to the :class:`~vectorstore.stores.base.VectorStore` holding that
    space's vectors; search always resolves the space first (via the
    router's selection), then queries only that space's store with that
    space's query embedding, so cross-space comparison is structurally
    impossible.

    Degradation rules:

    - No usable dense provider or store: lexical + structured still serve.
    - Primary provider fails mid-request: the failure is recorded (feeding
      the circuit breaker) and the fallback space is tried within the same
      request. The reverse never happens — when the router already rejected
      the primary (budget, breaker, config), a fallback failure ends the
      dense branch rather than overriding policy.
    - Lexical index unavailable: dense + structured still serve.
    """

    def __init__(
        self,
        catalog: DocumentCatalog,
        stores: Mapping[str, VectorStore] | None = None,
        router: EmbeddingRouter | None = None,
        analyzer: QueryAnalyzer | None = None,
        observer: RetrievalTraceObserver | None = None,
        config: RetrieverConfig | None = None,
    ) -> None:
        self._catalog = catalog
        self._stores = dict(stores or {})
        self._router = router
        self._analyzer = analyzer or QueryAnalyzer()
        self._observer = observer
        self._config = config or RetrieverConfig()
        if self._config.dense_enabled and self._router is not None:
            self._validate_spaces()

    def _validate_spaces(self) -> None:
        """Fail fast on store/spec mismatches instead of at query time."""

        assert self._router is not None
        providers = [self._router.primary]
        if self._router.fallback is not None:
            providers.append(self._router.fallback)
        for provider in providers:
            spec = provider.spec
            store = self._stores.get(spec.space_id)
            if store is None:
                continue
            if store.dimension is not None and store.dimension != spec.dimension:
                raise ValueError(
                    f"store registered for space {spec.space_id!r} expects "
                    f"{store.dimension}-dimensional vectors but the space "
                    f"produces {spec.dimension}; one store per embedding space"
                )

    @property
    def config(self) -> RetrieverConfig:
        return self._config

    def find(
        self,
        filter: MetadataFilter | None = None,
        scope: RetrievalScope | None = None,
        limit: int = 100,
    ) -> list[CatalogDocument]:
        """Structured-only retrieval, delegated to the catalog.

        This is the short-circuit for pure-filter requests; it involves no
        embeddings, no lexical search, and no fusion.
        """

        return self._catalog.find(filter=filter, scope=scope, limit=limit)

    def retrieve(
        self,
        query: str,
        *,
        filter: MetadataFilter | None = None,
        scope: RetrievalScope | None = None,
        k: int | None = None,
    ) -> RetrievalResult:
        """Hybrid retrieval: dense + lexical candidates fused with weighted RRF.

        ``scope`` is enforced inside candidate generation on both branches
        (SQL for lexical, filter pushdown for dense), never by
        post-filtering. ``k`` overrides the configured ``final_top_k``.
        """

        profile = self._analyzer.analyze(query)
        if profile.kind is QueryKind.EMPTY:
            raise ValueError(
                "query must be a non-empty string; use find() for "
                "structured-only retrieval"
            )
        final_k = k if k is not None else self._config.final_top_k
        if final_k <= 0:
            raise ValueError("k must be greater than zero")

        started = time.perf_counter()
        errors: list[str] = []

        dense_wanted = (
            self._config.dense_enabled
            and self._router is not None
            and bool(self._stores)
        )
        dense_ms: float | None = None
        dense: _DenseOutcome | None = None
        if dense_wanted:
            dense_started = time.perf_counter()
            dense = self._dense_search(query, merge_scope_filter(filter, scope))
            dense_ms = (time.perf_counter() - dense_started) * 1000
            errors.extend(dense.errors)

        lexical_ms: float | None = None
        lexical_hits: list[RankedHit] = []
        lexical_failed = False
        if self._config.lexical_enabled:
            lexical_started = time.perf_counter()
            try:
                lexical_hits = self._catalog.search_lexical(
                    query,
                    k=self._config.lexical_top_k,
                    filter=filter,
                    scope=scope,
                )
            except LexicalUnavailableError as exc:
                lexical_failed = True
                errors.append(f"lexical unavailable: {exc}")
            lexical_ms = (time.perf_counter() - lexical_started) * 1000

        dense_results = dense.results if dense is not None else []
        hits = self._fuse(dense_results, lexical_hits, profile, final_k)

        degraded = (
            dense_wanted and (dense is None or not dense.attempted)
        ) or lexical_failed

        selection = dense.selection if dense is not None else None
        result = RetrievalResult(
            hits=tuple(hits),
            query_id=uuid.uuid4().hex,
            query_kind=profile.kind,
            provider=selection.spec.provider if selection else None,
            model=selection.spec.model if selection else None,
            space_id=selection.spec.space_id if selection else None,
            provider_reason=str(selection.reason) if selection else None,
            fallback_occurred=dense.fallback_occurred if dense else False,
            degraded=degraded,
            dense_candidates=len(dense_results),
            lexical_candidates=len(lexical_hits),
            dense_weight=profile.dense_weight,
            lexical_weight=profile.lexical_weight,
            timings=RetrievalTimings(
                dense_ms=dense_ms,
                lexical_ms=lexical_ms,
                total_ms=(time.perf_counter() - started) * 1000,
            ),
            errors=tuple(errors),
        )
        self._notify(result)
        return result

    # -- dense branch -------------------------------------------------------

    def _dense_search(
        self, query: str, filter: MetadataFilter | None
    ) -> _DenseOutcome:
        assert self._router is not None
        errors: list[str] = []

        try:
            selection = self._router.select("query", texts=[query])
        except NoProviderAvailableError as exc:
            return _DenseOutcome(
                results=[],
                selection=None,
                fallback_occurred=False,
                attempted=False,
                errors=(f"dense unavailable: {exc}",),
            )

        candidates = [selection]
        # Fall forward primary -> fallback only. When the router already
        # rejected the primary (budget/breaker/config), retrying it here
        # would override policy.
        is_primary = selection.provider is self._router.primary
        if is_primary and self._router.fallback is not None:
            candidates.append(
                ProviderSelection(
                    self._router.fallback,
                    self._router.fallback.spec,
                    SelectionReason.OPENAI_UNAVAILABLE,
                )
            )

        for attempt, candidate in enumerate(candidates):
            store = self._stores.get(candidate.spec.space_id)
            if store is None:
                errors.append(
                    f"no vector store registered for space {candidate.spec.space_id!r}"
                )
                continue

            candidate_is_primary = candidate.provider is self._router.primary
            try:
                vector = candidate.provider.embed_query(query)
            except Exception as exc:  # noqa: BLE001 - degrade, never fail retrieval
                if candidate_is_primary:
                    self._router.record_failure()
                errors.append(
                    f"embedding failed ({candidate.spec.space_id}): {exc}"
                )
                continue
            if candidate_is_primary:
                self._router.record_usage(estimate_tokens([query]))

            try:
                results = store.search(
                    vector, k=self._config.dense_top_k, filter=filter
                )
            except Exception as exc:  # noqa: BLE001 - degrade, never fail retrieval
                errors.append(
                    f"vector search failed ({candidate.spec.space_id}): {exc}"
                )
                continue

            return _DenseOutcome(
                results=results,
                selection=candidate,
                fallback_occurred=candidate.is_fallback or attempt > 0,
                attempted=True,
                errors=tuple(errors),
            )

        return _DenseOutcome(
            results=[],
            selection=None,
            fallback_occurred=False,
            attempted=False,
            errors=tuple(errors),
        )

    # -- fusion and hydration -----------------------------------------------

    def _fuse(
        self,
        dense_results: list[SearchResult],
        lexical_hits: list[RankedHit],
        profile: QueryProfile,
        final_k: int,
    ) -> list[RetrievalHit]:
        dense_ids = [result.chunk.id for result in dense_results]
        lexical_ids = [hit.chunk_id for hit in lexical_hits]
        fused = rrf(
            [dense_ids, lexical_ids],
            weights=[profile.dense_weight, profile.lexical_weight],
            k=self._config.rrf_k,
        )[:final_k]

        dense_by_id = {
            result.chunk.id: (position, result.score)
            for position, result in enumerate(dense_results, start=1)
        }
        lexical_by_id = {hit.chunk_id: hit for hit in lexical_hits}

        chunk_ids = [chunk_id for chunk_id, _ in fused]
        chunks_by_id = {
            chunk.chunk_id: chunk for chunk in self._catalog.get_chunks(chunk_ids)
        }

        hits: list[RetrievalHit] = []
        for chunk_id, score in fused:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                # Dense stores can hold vectors for chunks the catalog no
                # longer knows (or never knew); those cannot be hydrated or
                # scope-verified, so they are dropped.
                continue
            dense_entry = dense_by_id.get(chunk_id)
            lexical_entry = lexical_by_id.get(chunk_id)
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=score,
                    dense_rank=dense_entry[0] if dense_entry else None,
                    dense_score=dense_entry[1] if dense_entry else None,
                    lexical_rank=lexical_entry.rank if lexical_entry else None,
                    lexical_score=lexical_entry.score if lexical_entry else None,
                )
            )
        return hits

    def _notify(self, result: RetrievalResult) -> None:
        if self._observer is None:
            return
        try:
            self._observer.on_retrieve(result)
        except Exception:  # noqa: BLE001 - telemetry must never break retrieval
            pass
