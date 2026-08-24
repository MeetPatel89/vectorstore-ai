"""Weighted Reciprocal Rank Fusion over ranked candidate lists.

RRF combines rankings using only positions, never raw scores, so cosine
similarities and BM25 values — which live on incomparable scales — are
never summed. Each list contributes ``weight / (k + rank)`` for every
candidate it contains; candidates absent from a list contribute nothing
from it.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_RRF_K = 60


def rrf(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists into one list of ``(id, score)`` pairs.

    ``rankings`` are best-first ID lists (rank 1 is index 0). ``weights``
    scales each list's contribution and defaults to 1.0 for all lists.
    Duplicate IDs within one list keep their best (first) rank. The output
    is sorted by descending fused score with ties broken by ID, so fusion
    is fully deterministic.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings in length")
    for weight in weights:
        if weight < 0:
            raise ValueError("weights must not be negative")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[str] = set()
        for position, candidate_id in enumerate(ranking, start=1):
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("ranking entries must be non-empty strings")
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            scores[candidate_id] = (
                scores.get(candidate_id, 0.0) + weight / (k + position)
            )

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
