"""Phase 2: provider routing, budgets, circuit breaking, and fallback."""

from __future__ import annotations

import argparse
import sys

from _corpus import add_corpus_argument, load_documents, to_vector_chunks
from _providers import HashEmbedding, make_embedder

from vectorstore import (
    CircuitBreaker,
    EmbeddingProvider,
    EmbeddingRouter,
    InMemoryBudgetLedger,
    NumpyVectorStore,
    ProviderSelection,
    VectorIndex,
)

DEMO_QUERY = "users can't log in after certificate rotation"
DAILY_BUDGET_USD = 0.001
SIMULATED_SPEND_USD = 0.002
SIMULATED_RATE_USD_PER_MILLION = 1.0


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the provider-routing walkthrough."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_corpus_argument(parser)
    parser.add_argument(
        "--provider",
        choices=("hash", "openai", "local"),
        default="hash",
        help="primary embedding provider (default: %(default)s, fully offline)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Phase 2 provider-routing walkthrough."""
    args = parse_args()
    try:
        primary = make_embedder(args.provider)
    except (ImportError, ValueError) as exc:
        print(
            f"Could not create the {args.provider!r} provider: {exc}", file=sys.stderr
        )
        return 2

    # The fallback deliberately has a different model and dimension. Even when
    # the offline hash provider is also the primary, its vectors occupy a
    # distinct embedding space.
    fallback = HashEmbedding(dimension=96, model="blake2b-bow-fallback")

    print("PHASE 2 — PROVIDER ROUTING")
    print(f"Primary space:  {primary.spec.space_id}\n{primary.spec}")
    print(f"Fallback space: {fallback.spec.space_id}\n{fallback.spec}")

    normal_router = EmbeddingRouter(primary, fallback)
    normal = normal_router.select("query", texts=[DEMO_QUERY])
    _print_selection("1. Normal request", primary, normal)

    budget_ledger = InMemoryBudgetLedger()
    budget_router = EmbeddingRouter(
        primary,
        fallback,
        ledger=budget_ledger,
        daily_budget_usd=DAILY_BUDGET_USD,
        cost_per_million_tokens=SIMULATED_RATE_USD_PER_MILLION,
    )
    budget_router.record_usage(
        tokens=2_000,
        usd=SIMULATED_SPEND_USD,
    )
    over_budget = budget_router.select("query", texts=[DEMO_QUERY])
    print(
        f"\n   Simulated daily spend: ${budget_ledger.spent_today():.4f} "
        f"(budget: ${DAILY_BUDGET_USD:.4f})"
    )
    _print_selection("2. Daily budget exhausted", primary, over_budget)

    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=300.0)
    circuit_router = EmbeddingRouter(primary, fallback, breaker=breaker)
    print("\n   Simulating primary-provider failures:")
    for attempt in range(1, 4):
        circuit_router.record_failure()
        print(f"     failure {attempt}: circuit open={breaker.is_open}")
    circuit_open = circuit_router.select("query", texts=[DEMO_QUERY])
    _print_selection("3. Primary circuit open", primary, circuit_open)

    override_router = EmbeddingRouter(
        primary,
        fallback,
        override="force_fallback",
    )
    forced = override_router.select("query", texts=[DEMO_QUERY])
    _print_selection("4. Manual force_fallback override", primary, forced)

    _show_separate_spaces(
        primary,
        fallback,
        corpus=args.corpus,
        primary_selection=normal,
        fallback_selection=over_budget,
    )
    return 0


def _print_selection(
    label: str,
    primary: EmbeddingProvider,
    selection: ProviderSelection,
) -> None:
    role = "primary" if selection.provider is primary else "fallback"
    print(f"\n{label}")
    print(f"   selected: {role} ({selection.spec.provider}/{selection.spec.model})")
    print(f"   reason:   {selection.reason.name} ({selection.reason.value})")
    print(f"   space:    {selection.spec.space_id}")


def _show_separate_spaces(
    primary: EmbeddingProvider,
    fallback: EmbeddingProvider,
    *,
    corpus: Path,
    primary_selection: ProviderSelection,
    fallback_selection: ProviderSelection,
) -> None:
    print("\nPHASE 2 — ONE STORE PER EMBEDDING SPACE")
    chunks = to_vector_chunks(load_documents(corpus))
    stores: dict[str, NumpyVectorStore] = {}
    for provider in (primary, fallback):
        store = NumpyVectorStore(dimension=provider.spec.dimension)
        VectorIndex(provider, store).index(chunks)
        stores[provider.spec.space_id] = store

    for label, provider in (("Primary", primary), ("Fallback", fallback)):
        store = stores[provider.spec.space_id]
        print(
            f"{label:8} {provider.spec.space_id}: "
            f"{store.count()} chunks, {store.dimension} dimensions"
        )

    primary_store = stores[primary_selection.spec.space_id]
    fallback_store = stores[fallback_selection.spec.space_id]
    uses_primary_store = primary_store is stores[primary.spec.space_id]
    print(f"Normal routing uses primary store: {uses_primary_store}")
    print(
        "Budget fallback uses fallback store: "
        f"{fallback_store is stores[fallback.spec.space_id]}"
    )
    print(f"Stores are distinct objects: {primary_store is not fallback_store}")
    print("Fallback vectors never enter or get compared with the primary space.")


if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
    raise SystemExit(main())
